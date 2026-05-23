"""
Build shuffle_index.npy for a dataset directory.

Reads observation_joinid from the obs file, looks up the corresponding row
index in the source h5ad, and saves shuffle_index.npy.

Verifies all joinids exist in the h5ad and are unique before saving.

Usage:
  python3 build_shuffle_index.py <data_dir>
          [--ge-h5ads DIR] [--obs-file FILENAME]
"""

import argparse
import gzip
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def run(data_dir, ge_h5ads_dir, obs_file):
    data_dir     = Path(data_dir)
    ge_h5ads_dir = Path(ge_h5ads_dir)

    ds_df = pd.read_csv(data_dir / "dataset_ids.tsv", sep="\t",
                        header=None, names=["dataset_id", "source"])
    if len(ds_df) != 1:
        print(f"ERROR: build_shuffle_index.py only supports single-dataset dirs "
              f"({len(ds_df)} datasets found)")
        return
    ds_id     = ds_df["dataset_id"].iloc[0]
    h5ad_path = ge_h5ads_dir / f"{ds_id}.h5ad"
    print(f"dataset: {ds_id}")
    print(f"h5ad:    {h5ad_path}")

    # Find obs file
    if obs_file:
        obs_path = data_dir / obs_file
    else:
        obs_files = sorted(data_dir.glob("obs*.tsv.gz"))
        if not obs_files:
            print(f"ERROR: no obs*.tsv.gz found in {data_dir}")
            return
        obs_path = obs_files[0]
    print(f"obs file: {obs_path.name}")

    # Load h5ad observation_joinids
    print(f"Loading observation_joinid from {h5ad_path.name} ...")
    with h5py.File(h5ad_path, "r") as f:
        raw = f["obs"]["observation_joinid"]
        if "categories" in raw:
            cats = np.array([x.decode() if isinstance(x, bytes) else str(x)
                             for x in raw["categories"][:]])
            ge_jids = cats[raw["codes"][:]]
        else:
            ge_jids = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                for x in raw[:]])
    n_unique = len(np.unique(ge_jids))
    print(f"  {len(ge_jids):,} cells, {n_unique:,} unique")
    assert n_unique == len(ge_jids), "observation_joinid not unique in h5ad!"

    joinid_to_row = {jid: i for i, jid in enumerate(ge_jids)}

    # Load obs joinids
    with gzip.open(obs_path, "rt") as f:
        jids = pd.read_csv(f, sep="\t", usecols=["observation_joinid"])["observation_joinid"].values
    print(f"obs rows: {len(jids):,}")

    # Verify all joinids exist in h5ad
    missing = [j for j in jids if j not in joinid_to_row]
    if missing:
        print(f"ERROR: {len(missing):,} joinids not found in h5ad — aborting")
        return
    print(f"all joinids found in h5ad")

    # Build shuffle index
    rows = np.array([joinid_to_row[j] for j in jids], dtype=np.int64)
    n_unique_rows = len(np.unique(rows))
    print(f"row range: min={rows.min()}, max={rows.max()}")
    print(f"unique rows: {n_unique_rows:,} (expect {len(jids):,})")
    if n_unique_rows != len(jids):
        print(f"ERROR: duplicate row indices — obs joinids not unique!")
        return

    out = data_dir / "shuffle_index.npy"
    np.save(out, rows)
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Dataset directory")
    parser.add_argument("--ge-h5ads", default="/mnt/efsUCE/ge_h5ads",
                        help="Directory containing h5ad files (default: /mnt/efsUCE/ge_h5ads)")
    parser.add_argument("--obs-file", default=None,
                        help="Obs filename within data_dir (default: first obs*.tsv.gz)")
    args = parser.parse_args()

    run(args.data_dir, args.ge_h5ads, args.obs_file)
