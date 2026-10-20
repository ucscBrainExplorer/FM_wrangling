from __future__ import annotations

"""
Run UMAP on a UCE embedding matrix and save the 2D coordinates to disk.

Follows the same PCA-then-UMAP convention used elsewhere in the repo
(e.g. analysis/BG_human_msn/uce_umap.py): the raw embedding is first
reduced with PCA (default 50 components) for speed/stability, then UMAP
is run on the PCA embedding. Pass --n-pca 0 to skip the PCA step and run
UMAP directly on the raw embedding instead.

Accepts three kinds of input, auto-detected by file extension:
    .npy    - a plain numpy array (e.g. mouse_mammary_epi_noPal_uce.npy)
    .h5ad   - an AnnData file; array read from .X, or .obsm[<key>] if --key given
    .h5     - a generic HDF5 file; array read from the dataset named by --key
              (default "data", matching uce.h5 / RunKnnNRP's convention;
              use --key pca to run on a run_pca.py output instead)

Output is a plain .npy file (default: umap.npy) of shape (n_cells, n_components).
To plot the result, use plot_umap.py on the saved .npy file.

Usage:
    python run_umap.py --input mouse_mammary_epi_noPal_uce.h5 --key data \
        --output umap.npy

    python run_umap.py --input mouse_mammary_epi_noPal_uce.npy \
        --n-pca 0 --output umap.npy   # skip PCA, run UMAP on raw embedding
"""

import argparse
import os
import sys

import h5py
import numpy as np

# Block tensorflow so `import umap` skips ParametricUMAP: the system TF build
# requires AVX and crashes with "Illegal instruction" on non-AVX hosts.
sys.modules.setdefault("tensorflow", None)
import umap
from sklearn.decomposition import PCA


def load_array(input_path: str, key: str | None) -> np.ndarray:
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".npy":
        return np.load(input_path)

    if ext == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(input_path)
        X = adata.X if key is None else adata.obsm[key]
        return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    if ext in (".h5", ".hdf5"):
        key = key or "data"
        with h5py.File(input_path, "r") as f:
            if key not in f:
                raise KeyError(f"Dataset '{key}' not found in {input_path}; found: {list(f.keys())}")
            return np.asarray(f[key])

    raise ValueError(f"Unsupported input extension: {ext} (expected .npy, .h5ad, or .h5)")


def run_umap(
    X: np.ndarray,
    n_pca: int = 50,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int | None = 42,
    n_jobs: int = -1,
) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)

    if n_pca and n_pca > 0:
        print(f"Running PCA ({n_pca} components) before UMAP...")
        pca = PCA(n_components=n_pca, random_state=random_state)
        X = pca.fit_transform(X)
        print(f"  explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    print(
        f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
        f"n_components={n_components}, metric={metric})..."
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=True,
    )
    embedding = reducer.fit_transform(X)
    print(f"UMAP complete: {embedding.shape}")
    return embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to .npy, .h5ad, or .h5 input")
    parser.add_argument(
        "--key",
        default=None,
        help="obsm key (.h5ad) or dataset name (.h5, default 'data'); unused for .npy",
    )
    parser.add_argument("--output", default="umap.npy", help="Output .npy path (default: umap.npy)")
    parser.add_argument("--n-pca", type=int, default=50, help="PCA components before UMAP; 0 to skip (default: 50)")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed (default: 42). A seed forces UMAP to run single-threaded; pass -1 for no seed "
        "to use all cores (faster, but the layout is not reproducible).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Threads for UMAP (default: -1 = all cores). Only takes effect with --random-state -1.",
    )
    args = parser.parse_args()
    if args.random_state < 0:
        args.random_state = None

    X = load_array(args.input, args.key)
    print(f"Loaded input array with shape {X.shape} from {args.input}")

    embedding = run_umap(
        X,
        n_pca=args.n_pca,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    np.save(args.output, embedding)
    print(f"Saved UMAP coordinates to {args.output}")


if __name__ == "__main__":
    main()
