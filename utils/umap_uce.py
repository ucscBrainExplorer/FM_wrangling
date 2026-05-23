#!/usr/bin/env python3
"""Run UMAP on obsm['X_uce'] in a combined h5ad.

Usage:
    python umap_uce.py combined_uce.h5ad [--out combined_umap.h5ad]
                       [--neighbors 15] [--min-dist 0.1] [--seed 42]

Adds obsm['X_umap'] to the h5ad and also saves coordinates as a .npy
alongside the output file.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="UMAP on UCE embeddings.")
    parser.add_argument("input", type=Path, help="Input h5ad with obsm['X_uce']")
    parser.add_argument("--out", "-o", type=Path, default=None,
                        help="Output h5ad (default: input stem + _umap.h5ad)")
    parser.add_argument("--neighbors", type=int, default=15,
                        help="n_neighbors for UMAP (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="min_dist for UMAP (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: {args.input} not found.")

    out_h5ad = args.out or args.input.parent / (args.input.stem + "_umap.h5ad")
    out_npy  = out_h5ad.with_name(out_h5ad.stem + "_coords.npy")

    print(f"Reading {args.input} ...")
    adata = ad.read_h5ad(args.input)

    if "X_uce" not in adata.obsm:
        sys.exit(f"Error: no obsm['X_uce'] in {args.input}. Keys: {list(adata.obsm.keys())}")

    uce = adata.obsm["X_uce"]
    print(f"  {uce.shape[0]:,} cells  dim={uce.shape[1]}")

    try:
        from umap import UMAP
    except ImportError:
        sys.exit("Error: umap-learn not installed. Run: pip install umap-learn")

    print(f"Running UMAP (n_neighbors={args.neighbors}, min_dist={args.min_dist}) ...")
    reducer = UMAP(
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        verbose=True,
    )
    coords = reducer.fit_transform(uce)

    adata.obsm["X_umap"] = coords

    print(f"Writing {out_h5ad} ...")
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_h5ad)

    print(f"Writing {out_npy} ...")
    np.save(out_npy, coords)

    # --- plot ---
    out_png = out_h5ad.with_suffix(".png")
    print(f"Plotting {out_png} ...")
    source_col = "source" if "source" in adata.obs.columns else None
    rng = np.random.default_rng(args.seed)
    shuffle = rng.permutation(len(coords))  # randomise render order

    fig, ax = plt.subplots(figsize=(8, 7))
    if source_col:
        labels = adata.obs[source_col].values[shuffle]
        xy = coords[shuffle]
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(xy[mask, 0], xy[mask, 1], s=0.5, alpha=0.5,
                       linewidths=0, label=label)
        ax.legend(markerscale=8, frameon=False, loc="upper right")
    else:
        ax.scatter(coords[shuffle, 0], coords[shuffle, 1],
                   s=0.5, alpha=0.5, linewidths=0, color="steelblue")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(out_h5ad.stem)
    ax.axis("equal")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Done.  UMAP range x=[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]"
          f"  y=[{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")


if __name__ == "__main__":
    main()
