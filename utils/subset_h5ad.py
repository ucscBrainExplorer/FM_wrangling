#!/usr/bin/env python3
"""Randomly subset cells from an h5ad file.

Usage:
    python subset_h5ad.py <input.h5ad> <n_cells> [--seed SEED] [--outdir DIR]

Output files (same directory as input, or --outdir):
    {basename}_{N}K.h5ad      — subset AnnData
    {basename}_{N}K_idx.npy   — integer row indices into the original file
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import anndata as ad


def human_label(n: int) -> str:
    """Format cell count as a compact label: 50000 → '50K', 1500000 → '1.5M'."""
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3g}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}K"
    if n >= 1_000:
        return f"{n / 1_000:.3g}K"
    return str(n)


def subset_h5ad(input_path: Path, n_cells: int, seed: int, outdir: Path) -> None:
    print(f"Reading {input_path} ...")
    with h5py.File(input_path, "r") as f:
        fmt = f["X"].attrs.get("encoding-type", "unknown")
    if fmt != "csr_matrix":
        print(f"WARNING: X is stored as '{fmt}', not csr_matrix. "
              "Row subsetting will load the full matrix into memory.")
    adata = ad.read_h5ad(input_path, backed="r")
    total = adata.n_obs

    if n_cells >= total:
        print(f"Requested {n_cells:,} >= total {total:,}; writing full dataset.")
        n_cells = total

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=n_cells, replace=False))

    label = human_label(n_cells)
    stem = input_path.stem
    out_h5ad = outdir / f"{stem}_{label}.h5ad"
    out_idx  = outdir / f"{stem}_{label}_idx.npy"

    print(f"Subsetting {n_cells:,} / {total:,} cells (seed={seed}) ...")
    # Write directly from the backed view — anndata reads only the selected
    # rows from the HDF5 file during the write, instead of loading the full
    # matrix into RAM first (which to_memory() would do).
    print(f"Writing {out_h5ad} ...")
    adata[idx].write_h5ad(out_h5ad)
    adata.file.close()

    print(f"Writing {out_idx} ...")
    np.save(out_idx, idx)

    print("Done.")
    print(f"  subset : {out_h5ad}  ({out_h5ad.stat().st_size / 1e9:.2f} GB)")
    print(f"  indices: {out_idx}")


def main():
    parser = argparse.ArgumentParser(description="Randomly subset cells from an h5ad file.")
    parser.add_argument("input",   type=Path, help="Input .h5ad file")
    parser.add_argument("n_cells", type=int,  help="Number of cells to sample")
    parser.add_argument("--seed",   type=int, default=42,  help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory (default: same as input file)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: {args.input} not found.")

    outdir = args.outdir if args.outdir else args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)

    subset_h5ad(args.input, args.n_cells, args.seed, outdir)


if __name__ == "__main__":
    main()
