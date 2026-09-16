from __future__ import annotations

"""
Run graph-based (Leiden or Louvain) clustering on an embedding matrix and
save the per-cell cluster labels to disk.

Follows the same neighbors-graph convention used by scanpy's standard
pipeline (PCA -> neighbors -> leiden): the raw embedding is optionally
reduced with PCA (default 50 components, same as run_umap.py), a fuzzy
simplicial-set neighbor graph is built (umap-learn's fuzzy_simplicial_set,
the same routine scanpy uses under the hood), and Leiden (default) or
Louvain community detection is run on that graph.

To avoid recomputing neighbors, pass --knn-cache with the .npz produced by
run_knn_ivfflat.py / build_knn_cache.py (indices + distances arrays); the
graph is then built directly from those cached neighbors instead of running
a fresh search, and --input / --n-pca are ignored (--metric still applies,
since it's passed to fuzzy_simplicial_set either way). --n-neighbors may
still be used to truncate the cache to its top-N nearest neighbors (the
cached neighbors are assumed sorted by increasing distance, as FAISS
search results are); omit it to use the full cached neighbor count.

Accepts three kinds of input, auto-detected by file extension (same as
run_umap.py / run_pca.py):
    .npy    - a plain numpy array (e.g. mouse_mammary_epi_noPal_uce.npy)
    .h5ad   - an AnnData file; array read from .X, or .obsm[<key>] if --key given
    .h5     - a generic HDF5 file; array read from the dataset named by --key
              (default "data", matching uce.h5; use --key pca for a
              run_pca.py output)

Output is a single-column table of integer cluster labels in row order
matching the input (default: clusters.tsv, a human-readable 'cluster'
column); pass --output clusters.npy for a plain numpy array instead.

Pass --obs with a TSV whose row order matches the input/knn-cache (first
column = cell barcode, e.g. the same obs file the embedding/knn-cache was
built from) to also emit a 'cell_id' column alongside 'cluster' (only for
.tsv/.csv output). This lets add_array_column.py do a real barcode-verified
merge downstream instead of trusting row order.

Usage:
    python run_clustering.py --input mouse_mammary_epi_noPal_uce.npy \
        --resolution 1.0 --output clusters.tsv

    # reuse an existing knn cache instead of recomputing neighbors
    python run_clustering.py --knn-cache knn_30_ivfflat.npz \
        --resolution 1.0 --output clusters.tsv

    # Louvain instead of Leiden
    python run_clustering.py --input mouse_mammary_epi_noPal_uce.npy \
        --method louvain --output clusters.tsv
"""

import argparse
import os

import h5py
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import fuzzy_simplicial_set


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


def build_graph(
    X: np.ndarray | None,
    n_neighbors: int,
    metric: str,
    random_state: int,
    knn_indices: np.ndarray | None = None,
    knn_distances: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Build a fuzzy simplicial-set neighbor graph (scanpy/UMAP convention)."""
    if knn_indices is not None:
        n_obs = knn_indices.shape[0]
        n_neighbors = knn_indices.shape[1]
        dummy_X = np.zeros((n_obs, 1), dtype=np.float32)
        print(f"Building neighbor graph from cached knn (n_neighbors={n_neighbors})...")
        connectivities, _, _ = fuzzy_simplicial_set(
            dummy_X,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_distances,
        )
    else:
        print(f"Searching {n_neighbors} nearest neighbors (metric={metric})...")
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X)
        distances, indices = nn.kneighbors(X)
        print(f"Building neighbor graph (n_neighbors={n_neighbors})...")
        connectivities, _, _ = fuzzy_simplicial_set(
            X,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=indices,
            knn_dists=distances,
        )
    return connectivities.tocsr()


def graph_to_igraph(connectivities: sparse.csr_matrix):
    import igraph as ig

    graph = sparse.triu(connectivities, k=1).tocoo()
    edges = list(zip(graph.row.tolist(), graph.col.tolist()))
    return ig.Graph(n=connectivities.shape[0], edges=edges, edge_attrs={"weight": graph.data.tolist()})


def run_clustering(
    connectivities: sparse.csr_matrix,
    method: str = "leiden",
    resolution: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    graph = graph_to_igraph(connectivities)

    if method == "leiden":
        import leidenalg as la

        print(f"Running Leiden clustering (resolution={resolution})...")
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
        )
    elif method == "louvain":
        print(f"Running Louvain clustering (resolution={resolution})...")
        partition = graph.community_multilevel(weights="weight", resolution=resolution)
    else:
        raise ValueError(f"Unknown method: {method} (expected 'leiden' or 'louvain')")

    labels = np.array(partition.membership, dtype=np.int64)
    print(f"Clustering complete: {labels.max() + 1} clusters")
    return labels


def save_labels(labels: np.ndarray, output_path: str, cell_ids: np.ndarray | None = None) -> None:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".tsv", ".csv"):
        import pandas as pd

        data = {"cluster": labels}
        if cell_ids is not None:
            data = {"cell_id": cell_ids, **data}
        pd.DataFrame(data).to_csv(output_path, sep="\t" if ext == ".tsv" else ",", index=False)
    else:
        if cell_ids is not None:
            print(f"Warning: --obs barcodes given but output is {ext}, which can't carry them; dropping cell_id")
        np.save(output_path, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="Path to .npy, .h5ad, or .h5 input (ignored if --knn-cache given)")
    parser.add_argument(
        "--key",
        default=None,
        help="obsm key (.h5ad) or dataset name (.h5, default 'data'); unused for .npy",
    )
    parser.add_argument(
        "--knn-cache",
        default=None,
        help="Path to a .npz with 'indices'/'distances' arrays (from run_knn_ivfflat.py or "
        "build_knn_cache.py) to build the graph from, skipping neighbor search. "
        "When given, --input/--n-pca are ignored (--metric still applies); --n-neighbors "
        "truncates to the cache's top-N nearest neighbors instead of a fresh search.",
    )
    parser.add_argument(
        "--output",
        default="clusters.tsv",
        help="Output path (default: clusters.tsv). Format follows the extension: "
        ".tsv/.csv write a single 'cluster' column as text, anything else (e.g. .npy) "
        "saves a plain numpy array.",
    )
    parser.add_argument("--n-pca", type=int, default=50, help="PCA components before neighbor search; 0 to skip (default: 50)")
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="Neighbors for a fresh search via --input (default: 30). With --knn-cache, "
        "truncates to the top N of the cached neighbors (must be <= cached count); "
        "omit to use the full cached neighbor count.",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "manhattan", "cosine", "correlation", "chebyshev", "minkowski"],
        default="euclidean",
        help="Distance metric for neighbor search / graph weighting (default: euclidean). "
        "Must be valid for both sklearn.neighbors.NearestNeighbors and umap's "
        "fuzzy_simplicial_set.",
    )
    parser.add_argument("--method", choices=["leiden", "louvain"], default="leiden")
    parser.add_argument("--resolution", type=float, default=1.0, help="Clustering resolution (default: 1.0)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--obs",
        default=None,
        help="Optional obs TSV whose row order matches the input/knn-cache; its first column "
        "(cell barcode) is carried into the output as a 'cell_id' column (.tsv/.csv output only) "
        "for a barcode-verified downstream join.",
    )
    args = parser.parse_args()

    cell_ids = None
    if args.obs:
        import pandas as pd

        obs_for_ids = pd.read_csv(args.obs, sep="\t")
        cell_ids = obs_for_ids.iloc[:, 0].to_numpy()
        print(f"Loaded {len(cell_ids):,} cell barcodes from {args.obs}")

    if args.knn_cache:
        print(f"Loading cached knn from {args.knn_cache}")
        cache = np.load(args.knn_cache)
        knn_indices = cache["indices"]
        knn_distances = cache["distances"]
        cached_k = knn_indices.shape[1]

        if args.n_neighbors is not None and args.n_neighbors != cached_k:
            if args.n_neighbors > cached_k:
                parser.error(
                    f"--n-neighbors {args.n_neighbors} exceeds the {cached_k} neighbors "
                    f"available in {args.knn_cache}"
                )
            print(f"Truncating cached neighbors from {cached_k} to top {args.n_neighbors}")
            knn_indices = knn_indices[:, : args.n_neighbors]
            knn_distances = knn_distances[:, : args.n_neighbors]

        connectivities = build_graph(
            None,
            n_neighbors=args.n_neighbors or cached_k,
            metric=args.metric,
            random_state=args.random_state,
            knn_indices=knn_indices,
            knn_distances=knn_distances,
        )
    else:
        if not args.input:
            parser.error("--input is required unless --knn-cache is given")
        X = load_array(args.input, args.key)
        print(f"Loaded input array with shape {X.shape} from {args.input}")
        X = np.ascontiguousarray(X, dtype=np.float32)

        if args.n_pca and args.n_pca > 0:
            print(f"Running PCA ({args.n_pca} components) before neighbor search...")
            pca = PCA(n_components=args.n_pca, random_state=args.random_state)
            X = pca.fit_transform(X)
            print(f"  explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        connectivities = build_graph(
            X,
            n_neighbors=args.n_neighbors if args.n_neighbors is not None else 30,
            metric=args.metric,
            random_state=args.random_state,
        )

    labels = run_clustering(
        connectivities,
        method=args.method,
        resolution=args.resolution,
        random_state=args.random_state,
    )

    if cell_ids is not None and len(cell_ids) != len(labels):
        raise ValueError(
            f"--obs row count ({len(cell_ids):,}) != cluster label count ({len(labels):,}) -- "
            "the obs file must have the same row order as the input/knn-cache"
        )

    save_labels(labels, args.output, cell_ids=cell_ids)
    print(f"Saved cluster labels to {args.output}")


if __name__ == "__main__":
    main()
