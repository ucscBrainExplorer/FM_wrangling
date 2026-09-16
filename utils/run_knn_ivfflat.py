from __future__ import annotations

"""
Build a FAISS IndexIVFFlat index and run a k=30 nearest-neighbor search
("knn30") on an embedding matrix.

The standard input is the raw UCE embedding (uce.h5, an (N, 1280) matrix
stored under the "data" dataset) - this is what index_ivfflat.faiss is built
on for every dataset, and what FoundationModelBenchmarking/cell_labelling/
predict.py queries with. pca.h5 (the 50-d PCA from run_pca.py) belongs to the
separate PCA -> PUMAP track and is NOT the KNN input.

Follows the same index convention as FM_wrangling/RunKnnNRP/knn_model.py +
knn_search.py: IndexIVFFlat quantized with IndexFlatL2, nlist = sqrt(N),
searched with nprobe=20.

Accepts three kinds of input, auto-detected by file extension:
    .npy    - a plain numpy array (e.g. mouse_mammary_epi_noPal_uce.npy)
    .h5ad   - an AnnData file; array read from .X, or .obsm[<key>] if --key given
    .h5     - a generic HDF5 file. With --key, that dataset is read. Without
              --key, the dataset is auto-detected: "data" (uce.h5) is tried
              first, then "pca" (pca.h5).

Outputs, both written next to --output:
    <output>.npz (default: knn_30_ivfflat.npz) - compressed, containing:
        indices    (n_cells, k) int64   - neighbor row indices into the input array
        distances  (n_cells, k) float32 - squared L2 distances to each neighbor
    index_ivfflat.faiss - the trained FAISS IndexIVFFlat, always written to the
        same directory as --output

Usage:
    python run_knn_ivfflat.py --input <dataset_dir>/uce.h5 \
        --output <dataset_dir>/knn_30_ivfflat.npz --k 30

    python run_knn_ivfflat.py --input mouse_mammary_epi_noPal_uce.npy \
        --output knn_30_ivfflat.npz --k 30
"""

import argparse
import math
import os

import faiss
import h5py
import numpy as np


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
        with h5py.File(input_path, "r") as f:
            if key is None:
                for candidate in ("data", "pca"):
                    if candidate in f:
                        key = candidate
                        break
                else:
                    raise KeyError(
                        f"No --key given and neither 'data' nor 'pca' found in {input_path}; "
                        f"found: {list(f.keys())}"
                    )
            elif key not in f:
                raise KeyError(f"Dataset '{key}' not found in {input_path}; found: {list(f.keys())}")
            print(f"Reading .h5 dataset '{key}' from {input_path}")
            return np.asarray(f[key])

    raise ValueError(f"Unsupported input extension: {ext} (expected .npy, .h5ad, or .h5)")


def build_ivfflat_index(X: np.ndarray, batch_size: int = 100_000) -> faiss.Index:
    """Build and train a FAISS IndexIVFFlat index, then add all vectors."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    nlist = int(math.sqrt(n))  # number of clusters, matches knn_model.py convention

    print(f"Embedding shape: {X.shape}")
    print(f"Building IVFFlat index (nlist={nlist}, dim={d})...")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    print("Training...")
    index.train(X)
    print("Training complete.")

    print("Adding vectors...")
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        index.add(X[i:end])
        print(f"  added {i}-{end} / {n}")

    return index, X


def run_knn_search(index: faiss.Index, X: np.ndarray, k: int, nprobe: int = 20):
    print(f"Running FAISS KNN search (k={k}, nprobe={nprobe})...")
    index.nprobe = nprobe
    distances, indices = index.search(X, k)
    print("KNN search complete.")
    return distances, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to .npy, .h5ad, or .h5 input")
    parser.add_argument(
        "--key",
        default=None,
        help="Which array to read. For .h5ad: read .obsm[KEY]; omit to read .X. "
        "For .h5: read the HDF5 dataset named KEY; omit to try 'data', then 'pca'. "
        "Ignored for .npy.",
    )
    parser.add_argument("--output", default="knn_30_ivfflat.npz", help="Output .npz path (default: knn_30_ivfflat.npz)")
    parser.add_argument("--k", type=int, default=30, help="Number of nearest neighbors (default: 30)")
    parser.add_argument("--nprobe", type=int, default=20, help="Clusters probed at search time (default: 20)")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Batch size for adding vectors (default: 100000)")
    args = parser.parse_args()

    X = load_array(args.input, args.key)
    index, X = build_ivfflat_index(X, batch_size=args.batch_size)

    index_output = os.path.join(os.path.dirname(os.path.abspath(args.output)), "index_ivfflat.faiss")
    faiss.write_index(index, index_output)
    print(f"Saved FAISS index to {index_output}")

    distances, indices = run_knn_search(index, X, args.k, nprobe=args.nprobe)

    np.savez_compressed(args.output, indices=indices, distances=distances)
    print(f"Saved KNN{args.k} results to {args.output}  (indices {indices.shape}, distances {distances.shape})")


if __name__ == "__main__":
    main()
