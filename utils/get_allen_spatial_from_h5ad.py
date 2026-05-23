"""
Extract spatial coordinates from a local h5ad file and save to TSV.
Reads X_CCF (→ x_ccf, y_ccf, z_ccf) and X_spatial_coords (→ x_section, y_section) from obsm.

Usage:
    python FM_wrangling/get_spatial_from_h5ad.py \
        --h5ad   ge_h5ads/73c39ba5-17dc-4630-b99a-0d43164a3945.h5ad \
        --output UCE_npy_obs/73c39ba5-17dc-4630-b99a-0d43164a3945_spatial.tsv
"""
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument("--h5ad",   required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

print(f"Reading: {args.h5ad}")
adata = sc.read_h5ad(args.h5ad, backed="r")
print(f"  {adata.n_obs:,} cells, obsm keys: {list(adata.obsm.keys())}")

cols = {}
if "X_CCF" in adata.obsm:
    ccf = np.array(adata.obsm["X_CCF"])
    cols["x_ccf"], cols["y_ccf"], cols["z_ccf"] = ccf[:, 0], ccf[:, 1], ccf[:, 2]
    print(f"  X_CCF: {ccf.shape}, NaN rows: {int(np.isnan(ccf).any(axis=1).sum()):,}")

if "X_spatial_coords" in adata.obsm:
    sc2 = np.array(adata.obsm["X_spatial_coords"])
    cols["x_section"], cols["y_section"] = sc2[:, 0], sc2[:, 1]
    print(f"  X_spatial_coords: {sc2.shape}, NaN rows: {int(np.isnan(sc2).any(axis=1).sum()):,}")

if not cols:
    raise SystemExit("No spatial obsm keys found (expected X_CCF or X_spatial_coords)")

df = pd.DataFrame(cols, index=adata.obs.index)
print(f"Writing {len(df):,} rows to {args.output}")
df.to_csv(args.output, sep="\t")
print("Done.")
