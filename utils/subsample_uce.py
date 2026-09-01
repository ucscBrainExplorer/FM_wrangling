#!/usr/bin/env python3
"""
Subsample a single UCE npy/h5/obs.tsv pair.

Usage:
    python subsample_uce.py <uce.npy|uce.h5> <obs.tsv> <n> [--seed 42] [--out_dir .]

The embedding file may be a .npy array or a .h5 file with a single 'data'
dataset (the format written by npy_to_h5.py). The output is written in the
same format as the input.
"""

import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("emb", type=Path, help="UCE embedding file (.npy or .h5)")
parser.add_argument("obs", type=Path)
parser.add_argument("n", type=int, help="Number of cells to sample")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=Path, default=Path("."))
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

if args.emb.suffix == ".npy":
    emb = np.load(args.emb)
elif args.emb.suffix in (".h5", ".hdf5"):
    with h5py.File(args.emb, "r") as f:
        emb = f["data"][:]
else:
    raise ValueError(f"Unsupported embedding format: {args.emb.suffix} (expected .npy or .h5)")

obs = pd.read_csv(args.obs, sep="\t", index_col=0)

assert emb.shape[0] == len(obs), "embedding and obs.tsv row counts don't match"
n = min(args.n, emb.shape[0])

idx = np.sort(rng.choice(emb.shape[0], size=n, replace=False))
sub_emb = emb[idx]

stem = args.emb.stem.removesuffix("_uce")
args.out_dir.mkdir(parents=True, exist_ok=True)

if args.emb.suffix == ".npy":
    np.save(args.out_dir / f"{stem}_uce_{n}.npy", sub_emb)
else:
    out_path = args.out_dir / f"{stem}_uce_{n}{args.emb.suffix}"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=sub_emb, dtype="float32",
                          chunks=True, compression="gzip", compression_opts=4)

obs.iloc[idx].reset_index(drop=True).to_csv(args.out_dir / f"{stem}_obs_{n}.tsv", sep="\t")
np.save(args.out_dir / f"{stem}_index_{n}.npy", idx)

print(f"Sampled {n} / {emb.shape[0]} cells -> {args.out_dir}/")
