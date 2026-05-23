"""
Verify that a dataset directory's obs file and shuffle_index.npy are consistent
with the source ge h5ad files.

Checks:
  1. shuffle_index.npy (if present): values in range, no duplicates
  2. Cross-validates obs file rows against h5ad obs columns at sampled positions,
     mapping through shuffle_idx when present.  Catches missing/wrong shuffle index.
  3. For multi-dataset dirs: verifies obs file has continuous dataset_id blocks
     matching dataset_ids.tsv order and sizes.
  4. pred_obs row count matches shuffle_idx length (or total cells if unshuffled).

Column selection for cross-validation:
  - Finds columns present in both the obs file and h5ad obs group.
  - Ranks by cardinality (estimated from a sample); picks top columns up to
    --n-cols (default 3).  observation_joinid wins if present (near-unique).

Usage:
  python3 verify_dir.py <data_dir>
          [--ge-h5ads DIR] [--n-check 20] [--n-cols 3] [--subset]
"""

import argparse
import sys
import gzip
import h5py
import numpy as np
import pandas as pd
from pathlib import Path



def get_h5ad_obs_keys(h5_path):
    with h5py.File(h5_path, "r") as f:
        return set(f["obs"].keys())


def get_n_cells(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "obs" in f:
            # use first scalar obs key to get length
            for k in f["obs"]:
                v = f["obs"][k]
                if "codes" in v:
                    return len(v["codes"])
                return len(v)
    raise ValueError(f"Cannot determine cell count from {h5_path}")


def read_obs_cols(obs_path, cols):
    with gzip.open(obs_path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", usecols=cols, low_memory=False)
    return df


def pick_check_cols(obs_path, h5ad_keys, n_cols, sample=500):
    """Find cols present in both obs file and h5ad; rank by cardinality; return top n_cols."""
    with gzip.open(obs_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")

    common = [c for c in header if c in h5ad_keys and c != "new_index"]
    if not common:
        return []

    # prefer observation_joinid — skip cardinality ranking
    if "observation_joinid" in common:
        rest = [c for c in common if c != "observation_joinid"]
        # still pick best companions
        candidates = rest
    else:
        candidates = common

    if candidates:
        with gzip.open(obs_path, "rt") as fh:
            sample_df = pd.read_csv(fh, sep="\t", usecols=candidates,
                                    nrows=sample, low_memory=False)
        cardinalities = {c: sample_df[c].nunique() for c in candidates}
        ranked = sorted(candidates, key=lambda c: -cardinalities[c])
    else:
        ranked = []

    if "observation_joinid" in common:
        selected = ["observation_joinid"] + ranked[:n_cols - 1]
    else:
        selected = ranked[:n_cols]

    return selected


def verify_cross(obs_path, h5ad_path, shuffle_idx, check_cols, n_check, label,
                 max_obs_rows=2000):
    """Spot-check obs rows against h5ad at sampled positions via check_cols.

    Only reads the first max_obs_rows rows of the obs file — sufficient to detect
    wrong or missing shuffle index without loading the full file.
    """
    with gzip.open(obs_path, "rt") as fh:
        obs_df = pd.read_csv(fh, sep="\t", usecols=check_cols,
                             nrows=max_obs_rows, low_memory=False)
    n = len(obs_df)

    positions = np.unique(np.concatenate([
        np.arange(min(5, n)),
        np.linspace(0, n - 1, n_check, dtype=int),
        np.arange(max(0, n - 5), n),
    ])).astype(int)

    # compute h5ad rows needed; sort for h5py (requires increasing order), then invert
    h5_rows  = np.array([int(shuffle_idx[i]) if shuffle_idx is not None else i
                         for i in positions], dtype=np.int64)
    sort_ord = np.argsort(h5_rows)
    inv_ord  = np.argsort(sort_ord)
    h5_rows_sorted = h5_rows[sort_ord]

    # load only needed rows from h5ad columns
    h5ad_cols = {}
    with h5py.File(h5ad_path, "r") as f:
        for col in check_cols:
            grp = f["obs"]
            if col not in grp:
                print(f"  [cross-check] WARNING: column {col!r} not in h5ad — skipping")
                continue
            raw = grp[col]
            if "categories" in raw:
                cats = np.array([
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in raw["categories"][:]
                ])
                codes = raw["codes"][h5_rows_sorted][inv_ord]
                h5ad_cols[col] = cats[codes]
            else:
                vals = raw[h5_rows_sorted][inv_ord]
                h5ad_cols[col] = np.array([
                    x.decode() if isinstance(x, bytes) else str(x) for x in vals
                ])

    if not h5ad_cols:
        print(f"  [cross-check] no readable columns — skipping")
        return

    failures = []
    for idx, (i, h5_row) in enumerate(zip(positions, h5_rows)):
        for col, h5_vals in h5ad_cols.items():
            obs_val = str(obs_df[col].iloc[i])
            h5_val  = str(h5_vals[idx])
            if obs_val != h5_val:
                failures.append((i, h5_row, col, obs_val, h5_val))

    if failures:
        print(f"  [cross-check] FAIL — {len(failures)} mismatches ({label}):")
        for i, h5_row, col, ov, hv in failures[:5]:
            print(f"    obs[{i}].{col}={ov!r}  h5ad[{h5_row}].{col}={hv!r}")
        sys.exit(1)

    print(f"  [cross-check] OK — {len(positions)} positions, cols={check_cols} ({label})")


def verify_block_order(obs_path, shuffle_idx, dataset_ids, ds_sizes):
    """For multi-dataset dirs: verify obs dataset_id column has correct block structure."""
    obs_ds = pd.read_csv(obs_path, sep="\t", usecols=["dataset_id"],
                         low_memory=False)["dataset_id"].values

    if shuffle_idx is not None:
        inv      = np.argsort(shuffle_idx)
        obs_ds   = obs_ds[inv]

    boundaries  = np.where(np.concatenate([[True], obs_ds[1:] != obs_ds[:-1]]))[0]
    block_ids   = [obs_ds[b] for b in boundaries]
    block_sizes = np.diff(np.append(boundaries, len(obs_ds))).tolist()

    if len(block_ids) != len(dataset_ids):
        print(f"  [block-order] FAIL — {len(block_ids)} blocks but {len(dataset_ids)} datasets")
        print(f"    blocks: {block_ids}")
        print(f"    expected: {dataset_ids}")
        sys.exit(1)

    for i, (got_id, exp_id, got_n, exp_n) in enumerate(
            zip(block_ids, dataset_ids, block_sizes, ds_sizes)):
        if got_id != exp_id:
            print(f"  [block-order] FAIL — block {i} is {got_id!r}, expected {exp_id!r}")
            sys.exit(1)
        if got_n != exp_n:
            print(f"  [block-order] FAIL — block {i} ({got_id}) has {got_n:,} rows, h5ad has {exp_n:,}")
            sys.exit(1)

    label = "shuffled" if shuffle_idx is not None else "sequential"
    print(f"  [block-order] OK — {len(block_ids)} blocks match dataset_ids.tsv ({label})")


def run(data_dir, ge_h5ads_dir, n_check, n_cols, subset):
    data_dir     = Path(data_dir)
    ge_h5ads_dir = Path(ge_h5ads_dir)
    ok = True

    # --- shuffle index ---
    shuffle_path = data_dir / "shuffle_index.npy"
    if shuffle_path.exists():
        shuffle_idx = np.load(shuffle_path)
        print(f"shuffle_index.npy: {len(shuffle_idx):,} entries")
    else:
        shuffle_idx = None
        print("No shuffle_index.npy — treating as sequential")

    # --- dataset ids and sizes ---
    ds_df       = pd.read_csv(data_dir / "dataset_ids.tsv", sep="\t",
                               header=None, names=["dataset_id", "source"])
    dataset_ids = ds_df["dataset_id"].tolist()
    print(f"{len(dataset_ids)} dataset(s): {dataset_ids}")

    ds_sizes = []
    for ds_id in dataset_ids:
        h5_path = ge_h5ads_dir / f"{ds_id}.h5ad"
        n = get_n_cells(h5_path)
        ds_sizes.append(n)
        print(f"  {ds_id}: {n:,} cells")
    N = sum(ds_sizes)
    print(f"Total h5ad cells: {N:,}")

    # --- shuffle index sanity ---
    if shuffle_idx is not None:
        if subset:
            if shuffle_idx.max() >= N:
                print(f"[shuffle] FAIL — max index {shuffle_idx.max():,} >= N {N:,}")
                sys.exit(1)
        else:
            if len(shuffle_idx) != N:
                print(f"[shuffle] FAIL — len {len(shuffle_idx):,} != N {N:,}")
                sys.exit(1)
        n_uniq = len(np.unique(shuffle_idx))
        if n_uniq != len(shuffle_idx):
            print(f"[shuffle] FAIL — {len(shuffle_idx) - n_uniq:,} duplicate indices")
            sys.exit(1)
        print(f"[shuffle] OK — {len(shuffle_idx):,} entries, all unique, max={shuffle_idx.max():,} < {N:,}")

    # --- obs file ---
    obs_files = sorted(data_dir.glob("obs*.tsv.gz"))
    if not obs_files:
        print("WARNING: no obs*.tsv.gz found — skipping cross-check and block-order check")
    else:
        obs_path = obs_files[0]
        print(f"obs file: {obs_path.name}")

        # pick check columns from first h5ad
        h5ad_path = ge_h5ads_dir / f"{dataset_ids[0]}.h5ad"
        h5ad_keys = get_h5ad_obs_keys(h5ad_path)
        check_cols = pick_check_cols(obs_path, h5ad_keys, n_cols)
        if not check_cols:
            print("WARNING: no common columns between obs file and h5ad — skipping cross-check")
        else:
            label = "shuffled" if shuffle_idx is not None else "sequential"
            verify_cross(obs_path, h5ad_path, shuffle_idx, check_cols, n_check, label)

        # block order check (multi-dataset only)
        if len(dataset_ids) > 1:
            verify_block_order(obs_path, shuffle_idx, dataset_ids, ds_sizes)

    # --- pred_obs row count ---
    pred_path = data_dir / "prediction_obs.tsv.gz"
    if pred_path.exists():
        n_pred = sum(1 for _ in gzip.open(pred_path, "rt")) - 1  # subtract header
        N_obs  = len(shuffle_idx) if shuffle_idx is not None else N
        if n_pred != N_obs:
            print(f"[pred_obs] FAIL — {n_pred:,} rows, expected {N_obs:,}")
            sys.exit(1)
        print(f"[pred_obs] OK — {n_pred:,} rows")

    print("\nAll checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--ge-h5ads", default="/mnt/efsUCE/ge_h5ads")
    parser.add_argument("--n-check", type=int, default=200,
                        help="Number of positions to spot-check (default: 200)")
    parser.add_argument("--n-cols", type=int, default=3,
                        help="Number of columns to use for cross-validation (default: 3)")
    parser.add_argument("--subset", action="store_true",
                        help="shuffle_index.npy is a known subset of the h5ad (relaxes length check)")
    args = parser.parse_args()

    run(args.data_dir, args.ge_h5ads, args.n_check, args.n_cols, args.subset)
