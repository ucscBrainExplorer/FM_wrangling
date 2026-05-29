#!/usr/bin/env python3
"""
One-time FAISS search for all query cells — saves neighbor indices and distances
as a compact .npz file (~36 MB for 100K cells × k=30).

Downstream scripts (plot_neighbor_ccf.py, precompute_neighbor_ccf.py) load this
cache via --knn_cache to skip the FAISS index load and search entirely.

Usage:
    python3 build_knn_cache.py \
        --query_emb /path/to/BG_human_100K_uce.npy \
        --index     /path/to/index_ivfflat.faiss \
        --out       /path/to/knn_cache.npz \
        --k         30
"""

import argparse
import numpy as np
import faiss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query_emb", required=True, help="Query embeddings .npy (N × d float32)")
    p.add_argument("--index",     required=True, help="FAISS index file (.faiss)")
    p.add_argument("--out",       required=True, help="Output .npz path")
    p.add_argument("--k",         type=int, default=30, help="Number of neighbors (default: 30)")
    p.add_argument("--nprobe",    type=int, default=20, help="IVF nprobe (default: 20)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading query embeddings from {args.query_emb}...")
    emb = np.load(args.query_emb).astype(np.float32)
    n_cells, d = emb.shape
    print(f"  {n_cells} cells, {d}d")

    print(f"Loading FAISS index from {args.index}...")
    index = faiss.read_index(args.index)
    ivf = faiss.extract_index_ivf(index)
    if ivf is not None:
        ivf.nprobe = args.nprobe
    print(f"  {index.ntotal} vectors, {index.d}d, nprobe={args.nprobe}")

    if index.d != d:
        raise ValueError(f"Embedding dim ({d}) != index dim ({index.d})")

    print(f"Searching k={args.k} nearest neighbors for {n_cells} cells...")
    sq_dists, indices = index.search(emb, args.k)
    dists = np.sqrt(np.maximum(sq_dists, 0)).astype(np.float32)
    indices = indices.astype(np.int32)   # int32 sufficient for up to 2B ref cells

    size_mb = (indices.nbytes + dists.nbytes) / 1e6
    print(f"  Done. Cache size: {size_mb:.1f} MB")

    print(f"Saving to {args.out}...")
    np.savez_compressed(args.out, indices=indices, distances=dists,
                        k=np.int32(args.k), nprobe=np.int32(args.nprobe))
    print(f"  Saved. indices={indices.shape}, distances={dists.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
