#!/usr/bin/env python3
"""Check that <ATLAS_DIR>/production/ (transform.json and the files it names) maps uce.h5 rows onto umap_<tag>.npy,
the way the site's mapping backend and map_newdata.py do it: IPCA -> remove d_hat directions (if any) -> PUMAP -> swap.
Also checks prediction_obs.tsv.gz row count against uce.h5.

Usage: python3 $FMW/verify_production.py <ATLAS_DIR> <umap_npy> [n_rows]   (build-reference.md, Production mapping files)
"""
import json
import os
import sys

import h5py
import joblib
import numpy as np

sys.path.insert(0, os.environ.get("UMAP_PYTORCH", f"/public/groups/hausslerlab/people/{os.environ.get('USER')}/umap_pytorch"))   # your umap_pytorch clone
sys.modules["tensorflow"] = None          # courtyard has no AVX; umap-learn would import it
import torch
import umap_pytorch

atlas, umap_file = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
prod = os.path.join(atlas, "production")
cfg = json.load(open(os.path.join(prod, "transform.json")))
print("transform.json:", json.dumps(cfg))

for k in ("pumap_model_file", "pca_model_file", "d_hat"):
    if k in cfg and not os.path.exists(os.path.join(prod, cfg[k])):
        sys.exit(f"ERROR: {k} = {cfg[k]} not in production/")

with h5py.File(os.path.join(atlas, "uce.h5"), "r") as f:
    total = f["data"].shape[0]
    idx = np.sort(np.random.default_rng(1).choice(total, n, replace=False))
    X = f["data"][idx]
ref = np.load(os.path.join(atlas, umap_file), mmap_mode="r")
assert ref.shape[0] == total, f"{umap_file} has {ref.shape[0]} rows, uce.h5 {total}"
ref = np.asarray(ref[idx])

P = joblib.load(os.path.join(prod, cfg["pca_model_file"])).transform(X).astype(np.float32)
if cfg.get("d_hat"):
    W = np.load(os.path.join(prod, cfg["d_hat"])).astype(np.float32)
    W = W.reshape(1, -1) if W.ndim == 1 else W
    P = P - (P @ W.T) @ W
    print(f"d_hat: {W.shape[0]} directions removed")
else:
    print("d_hat: none")
U = umap_pytorch.load_pumap(os.path.join(prod, cfg["pumap_model_file"])).transform(torch.from_numpy(P))
U = np.asarray(U) @ np.array(cfg["swap_xy_matrix"])
d = np.abs(U - ref).max(axis=1)
print(f"{n} random rows: max |diff| {d.max():.3g}, median {np.median(d):.3g} (map range {np.ptp(ref, axis=0)})")

nrow = sum(1 for _ in __import__("gzip").open(os.path.join(prod, "prediction_obs.tsv.gz"), "rt")) - 1
print(f"prediction_obs.tsv.gz: {nrow:,} rows, uce.h5 {total:,}")
ok = d.max() < 1e-3 and nrow == total
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
