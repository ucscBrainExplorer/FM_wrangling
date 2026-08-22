#!/usr/bin/env python3
"""
Take the first N rows of an already-shuffled UCE npy/h5 + obs.tsv pair.

For embeddings that were written in shuffled order (see build_shuffle_index.py /
*_shuffled dirs), the first N rows are already a random sample -- so this reads
one contiguous block instead of subset_uce.py's random point-selection. That
matters a lot on EFS/large files:
  - embedding: a scattered read over millions of rows means millions of small
    network reads; a sequential slice (mmap for .npy, HDF5 hyperslab for .h5)
    is one streamed read and never materializes the full array.
  - obs: obs.tsv(.gz) is read with nrows=n, so pandas stops parsing after the
    first N lines instead of decompressing/parsing the entire table (which for
    a wide obs with millions of rows can dominate both time and memory).

Usage:
    python head_uce.py <uce.npy|uce.h5> <obs.tsv[.gz]> <n> [--out_dir .]
"""

import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("emb", type=Path, help="UCE embedding file (.npy or .h5), already shuffled")
parser.add_argument("obs", type=Path)
parser.add_argument("n", type=int, help="Number of cells to take from the front")
parser.add_argument("--out_dir", type=Path, default=Path("."))
args = parser.parse_args()

# Get the embedding row count from metadata only -- no data read yet.
if args.emb.suffix == ".npy":
    emb = np.load(args.emb, mmap_mode="r")
    total = emb.shape[0]
elif args.emb.suffix in (".h5", ".hdf5"):
    emb = None  # reopened below for the actual slice read
    with h5py.File(args.emb, "r") as f:
        total = f["data"].shape[0]
else:
    raise ValueError(f"Unsupported embedding format: {args.emb.suffix} (expected .npy or .h5)")

n = min(args.n, total)

obs = pd.read_csv(args.obs, sep="\t", index_col=0, nrows=n)
assert len(obs) == n, f"{args.obs} has only {len(obs)} rows, need {n}"

if args.emb.suffix == ".npy":
    sub_emb = np.asarray(emb[:n])
else:
    with h5py.File(args.emb, "r") as f:
        sub_emb = f["data"][:n, :]

stem = args.emb.stem.removesuffix("_uce")
args.out_dir.mkdir(parents=True, exist_ok=True)

if args.emb.suffix == ".npy":
    np.save(args.out_dir / f"{stem}_uce_{n}.npy", sub_emb)
else:
    out_path = args.out_dir / f"{stem}_uce_{n}{args.emb.suffix}"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=sub_emb, dtype="float32",
                          chunks=True, compression="gzip", compression_opts=4)

obs.to_csv(args.out_dir / f"{stem}_obs_{n}.tsv", sep="\t")

print(f"Took first {n} / {total} cells (shuffled) -> {args.out_dir}/")
