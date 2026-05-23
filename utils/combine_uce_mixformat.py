#!/usr/bin/env python3
"""Combine UCE embeddings from multiple sources into a single h5ad.

Each input can be:
  - an h5ad file with obsm['X_uce']
  - a UCE .npy file  (obs is auto-found by replacing _uce.npy → _obs.tsv)

Usage:
    python combine_uce.py \\
        --input A_uce.npy \\
        --input B_50K_uce_adata.h5ad \\
        --label SpeciesA --label SpeciesB \\
        --out combined_uce.h5ad

The output h5ad contains only obs + obsm['X_uce'] (no X matrix).
A column 'source' is added to obs with the provided labels (or the file stem).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad


def load_source(path: Path, label: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (uce_matrix, obs_df) for one input, adding a 'source' column."""
    if path.suffix == ".h5ad":
        adata = ad.read_h5ad(path, backed="r")
        if "X_uce" not in adata.obsm:
            sys.exit(f"Error: {path} has no obsm['X_uce']. Keys: {list(adata.obsm.keys())}")
        uce = np.array(adata.obsm["X_uce"])
        obs = adata.obs.copy()
        adata.file.close()
    elif path.name.endswith("_uce.npy"):
        obs_path = Path(str(path).replace("_uce.npy", "_obs.tsv"))
        if not obs_path.exists():
            sys.exit(f"Error: expected obs file not found: {obs_path}")
        uce = np.load(path)
        obs = pd.read_csv(obs_path, sep="\t", index_col=0)
    else:
        sys.exit(f"Error: unrecognised input format: {path}\n"
                 "Expected .h5ad or *_uce.npy")

    obs = obs.copy()
    obs["source"] = label
    obs.index = obs.index.astype(str)
    print(f"  {label}: {uce.shape[0]:,} cells  dim={uce.shape[1]}")
    return uce, obs


def main():
    parser = argparse.ArgumentParser(description="Combine UCE embeddings into one h5ad.")
    parser.add_argument("--input",  "-i", action="append", required=True, metavar="FILE",
                        help="h5ad or *_uce.npy (repeat for each source)")
    parser.add_argument("--label",  "-l", action="append", metavar="LABEL",
                        help="Source label for each input (defaults to file stem)")
    parser.add_argument("--out",    "-o", required=True, type=Path, metavar="OUT.h5ad",
                        help="Output h5ad path")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]
    labels = args.label or []

    # pad missing labels with file stems
    while len(labels) < len(inputs):
        p = inputs[len(labels)]
        stem = p.name.replace("_uce.npy", "").replace(".h5ad", "")
        labels.append(stem)

    if len(inputs) < 2:
        sys.exit("Error: provide at least two --input files.")

    print("Loading sources ...")
    all_uce, all_obs = [], []
    for path, label in zip(inputs, labels):
        uce, obs = load_source(path, label)
        all_uce.append(uce)
        all_obs.append(obs)

    dims = [u.shape[1] for u in all_uce]
    if len(set(dims)) > 1:
        sys.exit(f"Error: embedding dimensions don't match: {dims}")

    print("Concatenating ...")
    combined_uce = np.concatenate(all_uce, axis=0)
    combined_obs = pd.concat(
        [obs.assign(_src_idx=i) for i, obs in enumerate(all_obs)],
        ignore_index=True,
    ).drop(columns="_src_idx")

    # h5py cannot write object columns with mixed types (e.g. bool + NaN).
    # Cast any remaining object columns to str, preserving numeric dtypes.
    for col in combined_obs.columns:
        if combined_obs[col].dtype == object:
            combined_obs[col] = combined_obs[col].where(
                combined_obs[col].isna(), combined_obs[col].astype(str)
            )

    out = ad.AnnData(obs=combined_obs)
    out.obsm["X_uce"] = combined_uce

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {args.out}  ({combined_uce.shape[0]:,} cells) ...")
    out.write_h5ad(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
