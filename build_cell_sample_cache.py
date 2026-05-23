"""
Build a per-dataset cell-sample cache for large h5ad files.

For each dataset with more than --max-cells cells, samples up to --max-cells
cells, extracts their gene expression (symbol genes only, no ribo/MT), bakes
in the annotation labels from prediction_obs, and saves a compact h5ad.

marker_genes.py auto-detects the cache and loads it instead of the full h5ad,
avoiding the ~95GB RAM spike on large datasets.

Cache saved as:  <data_dir>/<dataset_id>_sample.h5ad
  .X   -- counts from adata.X, sparse CSR, shape (n_sampled x n_symbol_genes)
  .obs -- annotation columns from prediction_obs for the sampled cells
  .var -- index = gene symbols (already filtered, no Ensembl / ribo / MT)

Usage:
  python3 build_cell_sample_cache.py <data_dir>
          [--ge-h5ads DIR] [--max-cells 250000] [--seed 42] [--force]
          [--datasets id1 id2 ...]   # process only these dataset IDs
"""

import argparse
import gc
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import h5py
from pathlib import Path


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

    boundaries = np.where(np.concatenate([[True], obs_ds[1:] != obs_ds[:-1]]))[0]
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


def _is_ribo_mt(g):
    u = g.upper()
    return u.startswith("RPS") or u.startswith("RPL") or u.startswith("MT-")


def _read_rows_csr(h5_path, x_path, sorted_rows, keep_col_idx, batch_size=5000):
    """Read specific rows from a CSR sparse matrix in an h5ad file.

    Processes rows in sorted batches for sequential I/O.  Only columns in
    keep_col_idx are retained.  Returns a CSR matrix (n_rows x len(keep_col_idx)).
    """
    sorted_rows  = np.asarray(sorted_rows, dtype=np.int64)
    n_out_rows   = len(sorted_rows)
    n_out_cols   = len(keep_col_idx)
    max_col      = int(keep_col_idx.max())
    col_remap    = np.full(max_col + 1, -1, dtype=np.int32)
    col_remap[keep_col_idx] = np.arange(n_out_cols, dtype=np.int32)

    # Group sampled rows into h5py read batches
    batch_ids    = sorted_rows // batch_size
    row_list, col_list, val_list = [], [], []

    with h5py.File(h5_path, "r") as f:
        grp    = f[x_path]
        indptr = grp["indptr"][:]          # (n_cells+1,) — cheap, ~11 MB for 1.4M cells

        prev_batch = -1
        raw_cols = raw_vals = io_start = None

        for out_i, r in enumerate(sorted_rows):
            b = int(batch_ids[out_i])
            if b != prev_batch:
                # Load a new contiguous block covering this batch
                bs  = b * batch_size
                be  = min(bs + batch_size, len(indptr) - 1)
                io_start = int(indptr[bs])
                io_end   = int(indptr[be])
                if io_end > io_start:
                    raw_cols = grp["indices"][io_start:io_end]
                    raw_vals = grp["data"][io_start:io_end]
                else:
                    raw_cols = raw_vals = np.array([], dtype=np.int32)
                prev_batch = b

            rs = int(indptr[r])     - io_start
            re = int(indptr[r + 1]) - io_start
            if rs == re:
                continue

            c = raw_cols[rs:re]
            v = raw_vals[rs:re]

            in_range = c <= max_col
            c_f      = c[in_range]
            v_f      = v[in_range]
            remapped = col_remap[c_f]
            keep     = remapped >= 0

            if keep.any():
                n_k = int(keep.sum())
                row_list.append(np.full(n_k, out_i, dtype=np.int32))
                col_list.append(remapped[keep])
                val_list.append(v_f[keep].astype(np.float32))

    if row_list:
        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
        vals = np.concatenate(val_list)
    else:
        rows = cols = np.array([], dtype=np.int32)
        vals = np.array([], dtype=np.float32)

    return sp.csr_matrix((vals, (rows, cols)),
                         shape=(n_out_rows, n_out_cols), dtype=np.float32)


def get_n_cells(path):
    with h5py.File(path, "r") as f:
        return int(f["X/indptr"].shape[0]) - 1 if "indptr" in f["X"] else int(f["X"].shape[0])


def get_batch_files(ge_h5ads_dir, ds_id):
    """Return sorted batch files for ds_id, or single file if no batches exist."""
    files = sorted(ge_h5ads_dir.glob(f"{ds_id}_???.h5ad"))
    return files if files else [ge_h5ads_dir / f"{ds_id}.h5ad"]


def _read_rows_from_batches(batch_files, x_path, sorted_rows, keep_col_idx, batch_size=5000):
    """Read sorted_rows spread across multiple batch h5ad files."""
    sizes   = [get_n_cells(p) for p in batch_files]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    parts   = []
    for b_path, b_start, b_end in zip(batch_files, offsets[:-1], offsets[1:]):
        in_b = (sorted_rows >= b_start) & (sorted_rows < b_end)
        if not in_b.any():
            continue
        b_rows = (sorted_rows[in_b] - b_start).astype(np.int64)
        parts.append(_read_rows_csr(b_path, x_path, b_rows, keep_col_idx, batch_size))
    if not parts:
        return sp.csr_matrix((0, len(keep_col_idx)))
    return sp.vstack(parts) if len(parts) > 1 else parts[0]


def build_gene_filter(var_df, var_names):
    if "feature_name" in var_df.columns:
        syms = var_df["feature_name"].values
        gene_names = np.where(
            pd.notna(syms) & (syms != "") & (syms != var_names),
            syms, var_names,
        )
    else:
        gene_names = var_names.copy()

    symbol_mask = ~np.array([g.startswith("ENSG") for g in gene_names])
    keep_mask   = symbol_mask & ~np.array([_is_ribo_mt(g) for g in gene_names])
    n_ensg      = (~symbol_mask).sum()
    n_ribo_mt   = symbol_mask.sum() - keep_mask.sum()
    keep_idx    = np.where(keep_mask)[0]
    filtered_names = gene_names[keep_mask]

    # Deduplicate: multiple Ensembl IDs can share a feature_name symbol; keep first occurrence
    seen = set()
    dedup_mask = np.zeros(len(filtered_names), dtype=bool)
    for i, g in enumerate(filtered_names):
        if g not in seen:
            seen.add(g)
            dedup_mask[i] = True
    n_dup = int((~dedup_mask).sum())

    return filtered_names[dedup_mask], keep_idx[dedup_mask], n_ensg, n_ribo_mt, n_dup


def _stratified_sample(local_rows, label_rows, pred_obs, ann_cols,
                        max_cells, min_cells, rng):
    """
    Iterative equal-share allocation stratified on the finest annotation column
    (the one with the most distinct groups).  Sorts groups by size ascending so
    small groups take what they need and leave the rest for larger ones.
    Cell types with < min_cells cells are not specially protected — they will
    be filtered by marker_genes.py anyway.
    Returns (sampled_local_sorted, sampled_label_sorted).
    """
    # Pick finest-grained column (most unique labels among valid cells)
    best_col, best_n = None, 0
    for col in ann_cols:
        vals  = pred_obs[col].values[label_rows]
        n_grp = len(np.unique(vals[pd.notna(vals)]))
        if n_grp > best_n:
            best_col, best_n = col, n_grp

    labels = pred_obs[best_col].values[label_rows]
    valid  = pd.notna(labels)
    unique, counts = np.unique(labels[valid], return_counts=True)

    # Drop groups too small to pass marker_genes.py filter — no point sampling them
    keep          = counts >= min_cells
    unique        = unique[keep]
    counts        = counts[keep]

    # Sort groups ascending by size
    order         = np.argsort(counts)
    sorted_groups = unique[order]
    sorted_counts = counts[order]

    # Iterative allocation
    allocation     = {}
    remaining_budget = min(max_cells, int(counts.sum()))
    remaining_groups = len(sorted_groups)
    for grp, cnt in zip(sorted_groups, sorted_counts):
        per_group        = remaining_budget // remaining_groups
        take             = min(int(cnt), per_group)
        allocation[grp]  = take
        remaining_budget -= take
        remaining_groups -= 1

    # Sample indices for each group
    all_indices = np.where(valid)[0]   # indices into label_rows / local_rows
    sampled_idx = []
    for grp, take in allocation.items():
        grp_idx = all_indices[labels[valid] == grp]
        if take >= len(grp_idx):
            sampled_idx.append(grp_idx)
        else:
            sampled_idx.append(rng.choice(grp_idx, size=take, replace=False))

    sampled_idx = np.concatenate(sampled_idx) if sampled_idx else np.array([], dtype=np.int64)

    # Sort by local row for sequential h5py reads
    sampled_local = local_rows[sampled_idx]
    sampled_label = label_rows[sampled_idx]
    sort_order    = np.argsort(sampled_local)
    return sampled_local[sort_order], sampled_label[sort_order]


def build_cache(ds_id, ds_size, ds_start, data_dir, ge_h5ads_dir,
                shuffle_idx, pred_obs, ann_cols, max_cells, min_cells, seed, force,
                row_chunk=50_000):

    out_path = data_dir / f"{ds_id}_sample.h5ad"

    if not force and out_path.exists():
        print(f"  cache exists — skip (--force to rebuild)")
        return

    if ds_size <= max_cells:
        print(f"  {ds_size:,} cells ≤ {max_cells:,} — no cache needed")
        return

    ds_end = ds_start + ds_size
    if shuffle_idx is not None:
        in_ds      = (shuffle_idx >= ds_start) & (shuffle_idx < ds_end)
        label_rows = np.where(in_ds)[0]
        local_rows = shuffle_idx[label_rows] - ds_start
    else:
        local_rows = np.arange(ds_size, dtype=np.int64)
        label_rows = np.arange(ds_start, ds_end, dtype=np.int64)

    rng = np.random.default_rng(seed)
    sampled_local_sorted, sampled_label_sorted = _stratified_sample(
        local_rows, label_rows, pred_obs, ann_cols, max_cells, min_cells, rng)
    print(f"  Sampled {len(sampled_local_sorted):,} / {ds_size:,} cells (stratified, min_cells={min_cells})")

    # Load gene metadata via backed mode (doesn't load X into RAM)
    batch_files = get_batch_files(ge_h5ads_dir, ds_id)
    h5_path     = batch_files[0]
    print(f"  Loading metadata from {h5_path.name} ({len(batch_files)} file(s)) ...")
    adata = ad.read_h5ad(h5_path, backed="r")
    print(f"    shape: {adata.shape}")

    var_df    = adata.var
    var_names = adata.var_names.values

    gene_names, keep_idx, n_ensg, n_ribo_mt, n_dup = build_gene_filter(var_df, var_names)
    n_genes = len(gene_names)
    print(f"    Genes kept: {n_genes:,}  (excluded {n_ensg:,} Ensembl, {n_ribo_mt:,} ribo/MT, {n_dup:,} dup symbols)")

    adata.file.close()
    del adata
    gc.collect()

    n_sampled = len(sampled_local_sorted)

    # Build obs/var upfront so we can write the skeleton before loading X
    obs_dict = {}
    for col in ann_cols:
        obs_dict[col] = pred_obs[col].values[sampled_label_sorted]
    obs_df = pd.DataFrame(obs_dict)
    var_df_out = pd.DataFrame(index=pd.Index(gene_names, name="gene_symbols"))

    # Write skeleton h5ad (empty X) to bake in correct obs/var encoding
    print(f"  Writing skeleton (obs/var) ...")
    empty_X = sp.csr_matrix((n_sampled, n_genes), dtype=np.float32)
    adata_skeleton = ad.AnnData(X=empty_X, obs=obs_df, var=var_df_out)
    adata_skeleton.write_h5ad(out_path)
    del adata_skeleton, empty_X
    gc.collect()

    # Fill X in row-chunks via h5py to keep peak memory bounded
    x_path = "X"
    print(f"  Filling X in {row_chunk:,}-row chunks via h5py ...")
    t0 = time.time()
    total_nnz = 0

    with h5py.File(out_path, "a") as f:
        del f["X"]
        xg = f.create_group("X")
        xg.attrs["encoding-type"] = "csr_matrix"
        xg.attrs["encoding-version"] = "0.1.0"
        xg.attrs["shape"] = np.array([n_sampled, n_genes], dtype=np.int64)

        ds_data    = xg.create_dataset("data",    shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1 << 20,))
        ds_indices = xg.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32,   chunks=(1 << 20,))
        ds_indptr  = xg.create_dataset("indptr",  shape=(1,), maxshape=(None,), dtype=np.int64,   chunks=(1 << 17,))
        ds_indptr[0] = 0

        cur_nnz = np.int64(0)
        cur_row = 0

        for chunk_start in range(0, n_sampled, row_chunk):
            chunk_local = sampled_local_sorted[chunk_start:chunk_start + row_chunk]
            nr = len(chunk_local)
            t_chunk = time.time()

            chunk_csr = _read_rows_from_batches(batch_files, x_path, chunk_local, keep_idx)
            nz = np.int64(chunk_csr.nnz)

            if nz > 0:
                new_nnz = int(cur_nnz + nz)
                ds_data.resize(new_nnz, axis=0)
                ds_data[int(cur_nnz):new_nnz] = chunk_csr.data
                ds_indices.resize(new_nnz, axis=0)
                ds_indices[int(cur_nnz):new_nnz] = chunk_csr.indices

            new_ip = chunk_csr.indptr[1:].astype(np.int64) + cur_nnz
            ds_indptr.resize(cur_row + nr + 1, axis=0)
            ds_indptr[cur_row + 1:cur_row + nr + 1] = new_ip

            cur_nnz += nz
            cur_row += nr
            total_nnz = int(cur_nnz)

            del chunk_csr
            gc.collect()
            print(f"    chunk {chunk_start // row_chunk + 1}: rows {chunk_start}–{chunk_start+nr-1}, nnz={int(nz):,}  ({time.time()-t_chunk:.0f}s)", flush=True)

    print(f"    shape: ({n_sampled}, {n_genes}), total nnz={total_nnz:,}  ({time.time()-t0:.0f}s)")
    print(f"  Saved: {out_path}  ({out_path.stat().st_size/1e9:.2f} GB)")


def run(data_dir, ge_h5ads_dir, max_cells, min_cells, seed, force, only_datasets, subset=False):
    data_dir     = Path(data_dir)
    ge_h5ads_dir = Path(ge_h5ads_dir)

    shuffle_path = data_dir / "shuffle_index.npy"
    if shuffle_path.exists():
        shuffle_idx = np.load(shuffle_path)
        print(f"Shuffle index found: {len(shuffle_idx):,} cells", flush=True)
    else:
        shuffle_idx = None
        print("No shuffle_index.npy — data treated as sequential (unshuffled).", flush=True)

    ds_df = pd.read_csv(data_dir / "dataset_ids.tsv", sep="\t",
                        header=None, names=["dataset_id", "source"])
    dataset_ids = ds_df["dataset_id"].tolist()
    print(f"  {len(dataset_ids)} datasets in cohort", flush=True)

    print(f"Reading cell counts from h5ads ...", flush=True)
    ds_sizes = []
    for ds_id in dataset_ids:
        n = sum(get_n_cells(p) for p in get_batch_files(ge_h5ads_dir, ds_id))
        ds_sizes.append(n)
        print(f"  {ds_id}: {n:,}", flush=True)
    N = sum(ds_sizes)
    if shuffle_idx is not None:
        if subset:
            assert shuffle_idx.max() < N, \
                f"shuffle index max {shuffle_idx.max():,} >= total cells {N:,}"
        else:
            assert len(shuffle_idx) == N, \
                f"Dataset sizes sum {N:,} != shuffle index length {len(shuffle_idx):,}"
        verify_shuffle_order(data_dir, shuffle_idx, dataset_ids)
    else:
        verify_sequential_order(data_dir, dataset_ids, ds_sizes)

    print(f"Loading prediction_obs ...", flush=True)
    pred_obs = pd.read_csv(data_dir / "prediction_obs.tsv.gz", sep="\t")
    N_obs = len(shuffle_idx) if (shuffle_idx is not None and subset) else N
    assert len(pred_obs) == N_obs, \
        f"prediction_obs rows {len(pred_obs):,} != expected {N_obs:,}"
    ann_cols = [c for c in pred_obs.columns if not c.endswith("_term_id")]
    print(f"Annotation columns: {ann_cols}", flush=True)

    ds_start = 0
    for ds_id, ds_size in zip(dataset_ids, ds_sizes):
        if only_datasets and ds_id not in only_datasets:
            ds_start += ds_size
            continue
        print(f"\n=== {ds_id} ({ds_size:,} cells) ===")
        build_cache(ds_id, ds_size, ds_start, data_dir, ge_h5ads_dir,
                    shuffle_idx, pred_obs, ann_cols, max_cells, min_cells, seed, force)
        ds_start += ds_size

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Shuffled dataset directory")
    parser.add_argument("--ge-h5ads", default="/mnt/efsUCE/ge_h5ads")
    parser.add_argument("--max-cells", type=int, default=250_000,
                        help="Max cells to sample per dataset (default: 250000)")
    parser.add_argument("--min-cells", type=int, default=500,
                        help="Match marker_genes.py --min-cells (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", nargs="+", metavar="ID",
                        help="Only process these dataset IDs")
    parser.add_argument("--subset", action="store_true",
                        help="Allow shuffle_index.npy to be a subset of the h5ad (relaxes length assertion)")
    args = parser.parse_args()

    run(args.data_dir, args.ge_h5ads, args.max_cells, args.min_cells,
        args.seed, args.force, set(args.datasets) if args.datasets else None,
        subset=args.subset)
