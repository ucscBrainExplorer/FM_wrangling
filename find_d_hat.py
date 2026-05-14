"""
Compute mean-diff direction (d_hat) between 10x v2 and v3 cells.
Restricts to cell types with >=1000 cells in both platforms.

Inputs:
  pca.h5       -- full PCA matrix, dataset key "pca"
  obs.tsv.gz   -- obs metadata with assay, suspension_type, cell_type columns

Output:
  meandiff_w.npy  -- (50,) unit vector, v2-v3 platform direction
"""

import numpy as np
import pandas as pd
import h5py

ID      = "ff9982c2-06cc-4376-930b-186fcbc1fd82"
DATA    = f"/mnt/efsUCE/HNOCA_Extended_{ID}"
OUT     = "/mnt/efsUCE/analysis/platform_viz_alignment/HNOCA/v2v3cell_shared/meandiff_w.npy"

CELL_LABEL_COL = "cell_type"   # swap to "harmonized_cell_label" if needed
MIN_CELLS      = 1000

print("Loading obs...")
obs = pd.read_csv(f"{DATA}/obs_harmonized_mapped_annotated.tsv.gz", sep="\t",
                  usecols=["assay", "suspension_type", CELL_LABEL_COL])
print(f"  total rows: {len(obs):,}")

v2_mask = (obs["assay"] == "10x 3' v2") & (obs["suspension_type"] == "cell")
v3_mask = (obs["assay"] == "10x 3' v3") & (obs["suspension_type"] == "cell")

v2_counts = obs.loc[v2_mask, CELL_LABEL_COL].value_counts()
v3_counts = obs.loc[v3_mask, CELL_LABEL_COL].value_counts()
shared = (set(v2_counts[v2_counts >= MIN_CELLS].index) &
          set(v3_counts[v3_counts >= MIN_CELLS].index))
shared.discard("unknown")
print(f"  shared cell types (>={MIN_CELLS} in both): {len(shared)}")

v2_idx = np.where(v2_mask & obs[CELL_LABEL_COL].isin(shared))[0]
v3_idx = np.where(v3_mask & obs[CELL_LABEL_COL].isin(shared))[0]
print(f"  v2: {len(v2_idx):,}   v3: {len(v3_idx):,}")

print("Loading PCA...")
with h5py.File(f"{DATA}/pca.h5") as f:
    pca_all = f["pca"][:]
print(f"  shape: {pca_all.shape}")

v2_pca = pca_all[v2_idx]
v3_pca = pca_all[v3_idx]

d = v2_pca.mean(0) - v3_pca.mean(0)
d_hat = (d / np.linalg.norm(d)).astype(np.float32)

np.save(OUT, d_hat)
print(f"Saved: {OUT}  {d_hat.shape}")
print("Done.")
