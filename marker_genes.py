"""
Marker gene analysis v2: Wilcoxon over all gene symbol genes, no HVG filtering.

Uses adata.X directly. Wilcoxon is rank-based and invariant to monotonic transforms.
Memory-efficient: processes genes in chunks, all groups scored per chunk.

Inputs (shuffled dataset directory):
  shuffle_index.npy      -- (N,) int64, shuffle_index[i] = original concat position of row i
  dataset_ids.tsv        -- tab-sep: dataset_id, source  (one row per dataset, concat order)
  prediction_obs.tsv.gz  -- (N rows) cell annotations; columns ending in _term_id are skipped

Outputs:
  <data_dir>/marker_genes.tsv        -- dataset_id, annotation_column, cell_type, rank, gene
  <data_dir>/marker_genes_docs.jsonl -- one doc per (dataset, annotation_column, cell_type)

Usage:
  python3 marker_genes_v2.py <data_dir> [--ge-h5ads DIR] [--n-top 30]
                             [--min-cells 50] [--chunk-size 500]
"""

import argparse
import json
import math
import gc
import time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import h5py
from pathlib import Path
from scipy.stats import rankdata


def sparse_rankdata(col):
    """Rank a 1-D array, exploiting zero sparsity for speed.
    Zeros are all tied and get rank (n_zeros+1)/2; nonzeros are ranked above them.
    Produces identical results to rankdata(col) with average tie handling.
    """
    n            = len(col)
    nonzero_mask = col != 0
    n_zeros      = n - nonzero_mask.sum()
    ranks        = np.empty(n, dtype=np.float64)
    if n_zeros > 0:
        ranks[~nonzero_mask] = (n_zeros + 1) / 2.0
    if n_zeros < n:
        ranks[nonzero_mask] = n_zeros + rankdata(col[nonzero_mask])
    return ranks


def get_n_cells(h5ad_path):
    with h5py.File(h5ad_path, "r") as f:
        if "indptr" in f["X"]:
            return int(f["X/indptr"].shape[0]) - 1
        return int(f["X"].shape[0])


def get_batch_files(ge_h5ads_dir, ds_id):
    files = sorted(ge_h5ads_dir.glob(f"{ds_id}_???.h5ad"))
    return files if files else [ge_h5ads_dir / f"{ds_id}.h5ad"]


def verify_shuffle_order(data_dir, shuffle_idx, dataset_ids):
    """Check that argsort(shuffle_idx) applied to an obs*.tsv.gz dataset_id column
    produces continuous blocks matching dataset_ids order. Raises if not."""
    obs_files = sorted(data_dir.glob("obs*.tsv.gz"))
    if not obs_files:
        print("  [shuffle check] no obs*.tsv.gz found — skipping order verification")
        return
    obs_path = obs_files[0]
    print(f"  [shuffle check] verifying order via {obs_path.name} ...")

    obs_ds = pd.read_csv(obs_path, sep="\t", usecols=["dataset_id"],
                         low_memory=False)["dataset_id"].values
    if len(obs_ds) != len(shuffle_idx):
        raise ValueError(
            f"shuffle_index length {len(shuffle_idx):,} != {obs_path.name} rows {len(obs_ds):,}")

    inv      = np.argsort(shuffle_idx)
    original = obs_ds[inv]

    # Detect blocks
    boundaries = np.where(np.concatenate([[True], original[1:] != original[:-1]]))[0]
    block_ids  = [original[b] for b in boundaries]

    if len(block_ids) != len(dataset_ids):
        raise ValueError(
            f"Reversed shuffle gives {len(block_ids)} blocks but dataset_ids.tsv has "
            f"{len(dataset_ids)} datasets.\nBlocks: {block_ids}\nExpected: {dataset_ids}")

    for i, (got, expected) in enumerate(zip(block_ids, dataset_ids)):
        if got != expected:
            raise ValueError(
                f"Block {i} is '{got}' but dataset_ids.tsv has '{expected}'. "
                f"Fix dataset_ids.tsv order to match obs file.")

    print(f"  [shuffle check] OK — {len(block_ids)} continuous blocks match dataset_ids.tsv")


def verify_sequential_order(data_dir, dataset_ids, ds_sizes):
    """For unshuffled data: check obs*.tsv.gz has continuous blocks matching
    dataset_ids order and sizes. Raises if not."""
    obs_files = sorted(data_dir.glob("obs*.tsv.gz"))
    if not obs_files:
        print("  [order check] no obs*.tsv.gz found — skipping order verification")
        return
    obs_path = obs_files[0]
    print(f"  [order check] verifying order via {obs_path.name} ...")

    obs_ds = pd.read_csv(obs_path, sep="\t", usecols=["dataset_id"],
                         low_memory=False)["dataset_id"].values

    boundaries  = np.where(np.concatenate([[True], obs_ds[1:] != obs_ds[:-1]]))[0]
    block_ids   = [obs_ds[b] for b in boundaries]
    block_sizes = np.diff(np.append(boundaries, len(obs_ds))).tolist()

    if len(block_ids) != len(dataset_ids):
        raise ValueError(
            f"obs file has {len(block_ids)} blocks but dataset_ids.tsv has "
            f"{len(dataset_ids)} datasets.\nBlocks: {block_ids}\nExpected: {dataset_ids}")

    for i, (got_id, exp_id, got_n, exp_n) in enumerate(
            zip(block_ids, dataset_ids, block_sizes, ds_sizes)):
        if got_id != exp_id:
            raise ValueError(
                f"Block {i} is '{got_id}' but dataset_ids.tsv has '{exp_id}'. "
                f"Fix dataset_ids.tsv order to match obs file.")
        if got_n != exp_n:
            raise ValueError(
                f"Block {i} ({got_id}) has {got_n:,} rows in obs file "
                f"but h5ad has {exp_n:,} cells.")

    print(f"  [order check] OK — {len(block_ids)} blocks, sizes and order match dataset_ids.tsv")


def run(data_dir, ge_h5ads_dir, n_top, min_cells, chunk_size, only_datasets=None, subset=False):
    data_dir     = Path(data_dir)
    ge_h5ads_dir = Path(ge_h5ads_dir)

    shuffle_path = data_dir / "shuffle_index.npy"
    shuffled     = shuffle_path.exists()
    if shuffled:
        shuffle_idx = np.load(shuffle_path)
        print("Shuffle index found.")
    else:
        shuffle_idx = None
        print("No shuffle_index.npy — data treated as sequential (unshuffled).")

    ds_df = pd.read_csv(data_dir / "dataset_ids.tsv", sep="\t",
                        header=None, names=["dataset_id", "source"])
    dataset_ids = ds_df["dataset_id"].tolist()

    ds_sizes = []
    for ds_id in dataset_ids:
        n = sum(get_n_cells(p) for p in get_batch_files(ge_h5ads_dir, ds_id))
        ds_sizes.append(n)
        print(f"  {ds_id}: {n:,} cells")

    N = sum(ds_sizes)
    print(f"Total cells: {N:,}")

    all_cached = all((data_dir / f"{ds_id}_sample.h5ad").exists() for ds_id in dataset_ids)

    if shuffled:
        if all_cached or subset:
            assert shuffle_idx.max() < N, \
                f"shuffle index max {shuffle_idx.max():,} >= total cells {N:,}"
        else:
            assert len(shuffle_idx) == N, \
                f"Dataset sizes sum {N:,} != shuffle index length {len(shuffle_idx):,}"
        verify_shuffle_order(data_dir, shuffle_idx, dataset_ids)
    else:
        verify_sequential_order(data_dir, dataset_ids, ds_sizes)

    pred_obs = pd.read_csv(data_dir / "prediction_obs.tsv.gz", sep="\t")
    N_obs = len(shuffle_idx) if (shuffled and (all_cached or subset)) else N
    assert len(pred_obs) == N_obs, \
        f"prediction_obs rows {len(pred_obs):,} != expected {N_obs:,}"
    ann_cols = [c for c in pred_obs.columns if not c.endswith("_term_id")]
    print(f"Annotation columns: {ann_cols}")

    docs_path = data_dir / "marker_genes.jsonl"
    docs_f = open(docs_path, "w")
    total_docs = 0

    ds_start = 0

    for ds_id, ds_size in zip(dataset_ids, ds_sizes):
        ds_end = ds_start + ds_size
        if only_datasets and ds_id not in only_datasets:
            ds_start = ds_end
            continue
        print(f"\n=== {ds_id} ({ds_size:,} cells) ===")

        cache_path = data_dir / f"{ds_id}_sample.h5ad"

        if cache_path.exists():
            # --- cache path: pre-sampled h5ad with obs already populated ---
            print(f"  Loading from cache: {cache_path.name}")
            adata_c    = ad.read_h5ad(cache_path)
            gene_names = adata_c.var_names.values.copy()
            n_genes    = len(gene_names)
            adata_shape = adata_c.shape
            cache_obs  = {col: adata_c.obs[col].values.copy()
                          for col in ann_cols if col in adata_c.obs.columns}
            X_ds       = adata_c.X
            del adata_c
            gc.collect()
            print(f"    shape: {adata_shape}, nnz={X_ds.nnz:,}")
            use_cache  = True
        else:
            # --- full h5ad path ---
            if shuffled:
                in_ds      = (shuffle_idx >= ds_start) & (shuffle_idx < ds_end)
                label_rows = np.where(in_ds)[0]
                local_rows = shuffle_idx[label_rows] - ds_start
                print(f"  Cells in shuffled data: {len(label_rows):,}")
            else:
                local_rows = np.arange(ds_size, dtype=np.int64)
                label_rows = np.arange(ds_start, ds_end, dtype=np.int64)
                print(f"  Cells: {ds_size:,}")

            print(f"  Loading h5ad ...")
            adata = ad.read_h5ad(ge_h5ads_dir / f"{ds_id}.h5ad")
            print(f"    shape: {adata.shape}")

            X_src     = adata.X
            var_names = adata.var_names.values
            var_df    = adata.var

            # Build gene name array: prefer feature_name symbols, fall back to var_names
            if "feature_name" in var_df.columns:
                symbols    = var_df["feature_name"].values
                gene_names = np.where(
                    pd.notna(symbols) & (symbols != "") & (symbols != var_names),
                    symbols, var_names
                )
            else:
                gene_names = var_names.copy()

            # Drop Ensembl ID fallbacks — UCE only accepts gene symbols
            symbol_mask = ~np.array([g.startswith("ENSG") for g in gene_names])

            # Drop ribosomal and mitochondrial genes (human/mouse/macaque, case-insensitive)
            def _is_ribo_mt(g):
                u = g.upper()
                return u.startswith("RPS") or u.startswith("RPL") or u.startswith("MT-")
            keep_mask      = symbol_mask & ~np.array([_is_ribo_mt(g) for g in gene_names])
            n_ensg         = (~symbol_mask).sum()
            n_ribo_mt      = symbol_mask.sum() - keep_mask.sum()
            symbol_col_idx = np.where(keep_mask)[0]
            gene_names     = gene_names[keep_mask]
            n_genes        = len(gene_names)
            print(f"    Genes kept: {n_genes:,}  (excluded {n_ensg:,} Ensembl, {n_ribo_mt:,} ribo/MT)")

            # Row slice first (CSR efficient), then free adata before column slice
            if not sp.isspmatrix_csr(X_src):
                X_src = sp.csr_matrix(X_src)
            X_tmp = X_src[local_rows]
            del adata, X_src
            gc.collect()
            X_ds  = X_tmp[:, symbol_col_idx].tocsc()   # column slice + convert (CSC for chunk access)
            del X_tmp
            gc.collect()
            print(f"    Matrix: {X_ds.shape[0]:,} × {X_ds.shape[1]:,}, nnz={X_ds.nnz:,}")
            use_cache = False

        for col in ann_cols:
            if use_cache:
                if col not in cache_obs:
                    print(f"  [{col}] not in cache obs — skip")
                    continue
                labels = cache_obs[col]
            else:
                labels = pred_obs[col].values[label_rows]
            valid  = pd.notna(labels)
            unique = np.unique(labels[valid])

            if len(unique) < 2:
                print(f"  [{col}] skip — <2 unique labels")
                continue

            min_cells_col = min_cells
            labels_valid  = labels[valid]

            group_names = [grp for grp in unique
                           if (labels_valid == grp).sum() >= min_cells_col]

            if len(group_names) < 2:
                print(f"  [{col}] skip — <2 groups with >= {min_cells_col} cells")
                continue

            print(f"  [{col}] {len(group_names)} groups (min_cells={min_cells_col:,})")

            valid_local = np.where(valid)[0]
            # Avoid a full CSC copy when all rows are valid (common case)
            X_valid = X_ds if len(valid_local) == X_ds.shape[0] else X_ds[valid_local]

            # Precompute group masks and Wilcoxon constants
            group_masks = [labels_valid == grp for grp in group_names]
            group_stats = []
            for mask in group_masks:
                n_grp  = mask.sum()
                n_rest = (~mask).sum()
                mu     = n_grp * n_rest / 2
                sigma  = math.sqrt(n_grp * n_rest * (n_grp + n_rest + 1) / 12)
                group_stats.append((mask, n_grp, mu, sigma))

            n_groups   = len(group_names)
            all_scores = np.full((n_groups, n_genes), -np.inf, dtype=np.float32)
            n_chunks   = math.ceil(n_genes / chunk_size)
            t0         = time.time()

            for chunk_i, chunk_start in enumerate(range(0, n_genes, chunk_size)):
                chunk_end = min(chunk_start + chunk_size, n_genes)
                chunk_len = chunk_end - chunk_start

                elapsed = time.time() - t0
                eta     = (elapsed / (chunk_i + 1)) * (n_chunks - chunk_i - 1) if chunk_i > 0 else 0
                print(f"    chunk {chunk_i+1}/{n_chunks}  genes {chunk_start}-{chunk_end}"
                      f"  elapsed {elapsed:.0f}s  eta {eta:.0f}s", flush=True)

                X_chunk     = X_valid[:, chunk_start:chunk_end].toarray().astype(np.float32)
                ranks_chunk = np.empty(X_chunk.shape, dtype=np.float32)
                for g in range(chunk_len):
                    ranks_chunk[:, g] = sparse_rankdata(X_chunk[:, g])

                for grp_i, (mask, n_grp, mu, sigma) in enumerate(group_stats):
                    rank_sum = ranks_chunk[mask].sum(axis=0)
                    U        = rank_sum - n_grp * (n_grp + 1) / 2
                    all_scores[grp_i, chunk_start:chunk_end] = (U - mu) / sigma

            print(f"    done in {time.time()-t0:.0f}s", flush=True)
            if X_valid is not X_ds:
                del X_valid
                gc.collect()

            for grp_i, grp in enumerate(group_names):
                top_idx  = np.argsort(-all_scores[grp_i])[:n_top]
                top_genes = [gene_names[i] for i in top_idx]

                n_grp = group_stats[grp_i][1]
                doc = {"dataset_id": ds_id, "annotation_column": col,
                       "cell_type": grp, "cell_count": int(n_grp),
                       "cell_fraction": round(n_grp / len(valid_local), 4),
                       "genes": top_genes}
                docs_f.write(json.dumps(doc) + "\n")
                docs_f.flush()
                total_docs += 1

                print(f"      saved: {grp} ({len(top_genes)} genes)", flush=True)

        del X_ds
        gc.collect()
        ds_start = ds_end

    docs_f.close()
    print(f"\nSaved: {docs_path}  ({total_docs:,} documents)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marker gene analysis v2 (all genes, raw counts)")
    parser.add_argument("data_dir",
                        help="Shuffled dataset directory")
    parser.add_argument("--ge-h5ads", default="/mnt/efsUCE/ge_h5ads",
                        help="Directory containing h5ad files (default: /mnt/efsUCE/ge_h5ads)")
    parser.add_argument("--n-top", type=int, default=30,
                        help="Top N DE genes per group (default: 30)")
    parser.add_argument("--min-cells", type=int, default=500,
                        help="Minimum cells per group (default: 500)")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Genes per chunk (default: 500)")
    parser.add_argument("--datasets", nargs="+", metavar="ID",
                        help="Only process these dataset IDs")
    parser.add_argument("--subset", action="store_true",
                        help="Allow shuffle_index.npy to be a subset of the h5ad (relaxes length assertion; only applies when cache is absent)")
    args = parser.parse_args()

    run(args.data_dir, args.ge_h5ads, args.n_top, args.min_cells, args.chunk_size,
        only_datasets=set(args.datasets) if args.datasets else None,
        subset=args.subset)
