#!/usr/bin/env python3
"""
Subsample a single UCE npy/obs.tsv pair.

Usage:
    python subsample_uce.py <uce.npy> <obs.tsv> <n> [--seed 42] [--out_dir .]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("npy", type=Path)
parser.add_argument("obs", type=Path)
parser.add_argument("n", type=int, help="Number of cells to sample")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=Path, default=Path("."))
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

emb = np.load(args.npy)
obs = pd.read_csv(args.obs, sep="\t", index_col=0)

assert emb.shape[0] == len(obs), "npy and obs.tsv row counts don't match"
n = min(args.n, emb.shape[0])

idx = np.sort(rng.choice(emb.shape[0], size=n, replace=False))

stem = args.npy.stem.removesuffix("_uce")
args.out_dir.mkdir(parents=True, exist_ok=True)
np.save(args.out_dir / f"{stem}_uce_{n}.npy", emb[idx])
obs.iloc[idx].reset_index(drop=True).to_csv(args.out_dir / f"{stem}_obs_{n}.tsv", sep="\t")
np.save(args.out_dir / f"{stem}_index_{n}.npy", idx)

print(f"Sampled {n} / {emb.shape[0]} cells -> {args.out_dir}/")
