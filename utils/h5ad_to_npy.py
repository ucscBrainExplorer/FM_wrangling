#!/usr/bin/env python3
"""
Convert UCE h5ad file to:
  - <stem>.npy       : UCE embeddings from adata.obsm['X_uce'], shape (n_cells, 1280)
  - <stem>_obs.tsv   : cell metadata from adata.obs, with cell barcodes as first column
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def convert(h5ad_path: str, output_dir=None, obsm_key: str = "X_uce") -> None:
    h5ad_path = Path(h5ad_path)
    out_dir = Path(output_dir) if output_dir else h5ad_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = h5ad_path.stem  # e.g. "uce"

    print(f"Reading {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  cells x genes : {adata.shape}")

    # --- embeddings → .npy ---
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in obsm. Available: {list(adata.obsm.keys())}")

    embeddings = adata.obsm[obsm_key]
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, embeddings)
    print(f"  Saved embeddings {embeddings.shape} ({embeddings.dtype}) → {npy_path}")

    # --- obs metadata → _obs.tsv ---
    obs: pd.DataFrame = adata.obs.copy()
    obs.index.name = "cell_barcode"
    obs_path = out_dir / f"{stem}_obs.tsv"
    obs.to_csv(obs_path, sep="\t")
    print(f"  Saved obs ({obs.shape[0]} cells, {obs.shape[1]} columns) → {obs_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", help="Input .h5ad file path")
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: same directory as input file)",
    )
    parser.add_argument(
        "--obsm-key",
        default="X_uce",
        help="Key in adata.obsm to save as .npy (default: X_uce)",
    )
    args = parser.parse_args()
    convert(args.h5ad, args.output_dir, args.obsm_key)


if __name__ == "__main__":
    main()
