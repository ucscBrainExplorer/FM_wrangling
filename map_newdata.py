import sys
import argparse

parser = argparse.ArgumentParser()
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--npy", help="UCE embeddings .npy file")
input_group.add_argument("--h5", help="UCE embeddings HDF5 file (dataset key: 'data')")
input_group.add_argument("--h5ad", help="UCE adata .h5ad file")
parser.add_argument("--obs", help="obs.tsv or obs.tsv.gz file (required when using --npy or --h5)")
parser.add_argument("--config", required=True, metavar="JSON", help="transform config, e.g. Wang2025Nature/transform_e100.json")
parser.add_argument("--out", default="newdata.arrow", help="output Arrow IPC file (default: newdata.arrow)")
parser.add_argument("--title", help="title stored in Arrow schema metadata")
parser.add_argument("--pin-first", metavar="COL", nargs="+", help="columns to place first in Arrow metadata, in order (e.g. --pin-first sample_label harmonized_cell_label)")
args = parser.parse_args()

if (args.npy or args.h5) and not args.obs:
    parser.error("--obs is required when using --npy or --h5")

import json
import joblib
import h5py
import torch
import numpy as np
import pandas as pd
import anndata
import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, "/mnt/efsUCE/umap_pytorch")  #git+https://github.com/ucscBrainExplorer/umap_pytorch.git
import umap_pytorch
print(umap_pytorch.__file__)

# Read JSON file
with open(args.config, "r") as f:
    config = json.load(f)

# Access fields
pca_model_file = config["pca_model_file"]
pumap_model_file = config["pumap_model_file"]
swap_xy_matrix = np.array(config["swap_xy_matrix"])  # convert to NumPy array

# read pca model back, pumap model back
ipca = joblib.load(pca_model_file)
pumap = umap_pytorch.load_pumap(pumap_model_file)

# read uce result back
if args.npy:
    uce = np.load(args.npy)
    obs = pd.read_csv(args.obs, sep="\t", index_col=0)
elif args.h5:
    with h5py.File(args.h5, "r") as f:
        uce = f["data"][:]
    obs = pd.read_csv(args.obs, sep="\t", index_col=0)
else:
    adata = anndata.read_h5ad(args.h5ad, backed='r')
    uce = adata.obsm["X_uce"]
    obs = adata.obs

# transform
X_pca = ipca.transform(uce)
X_pca.shape
X_pca = X_pca.astype(np.float32)

data_tensor = torch.from_numpy(X_pca)
X_umap_pytorch = pumap.transform(data_tensor)
X_umap_pytorch.shape

X_swapped = X_umap_pytorch @ swap_xy_matrix

# Auto-detect categorical columns: exclude those with 1 or >100 unique values
cat_cols = {}
for col in obs.columns:
    if hasattr(obs[col], 'dtype') and (
        str(obs[col].dtype) == 'category' or obs[col].dtype == object
    ):
        n_unique = obs[col].nunique()
        if 1 < n_unique <= 100:
            cat_cols[col] = pa.array(obs[col].astype("category"))

# Pin requested columns first, in order
if args.pin_first:
    pinned = {}
    for col in args.pin_first:
        if col not in cat_cols:
            print(f"WARNING: --pin-first column '{col}' not found in detected columns — ignoring")
        else:
            pinned[col] = cat_cols[col]
    rest = {k: v for k, v in cat_cols.items() if k not in pinned}
    cat_cols = {**pinned, **rest}

# Convert to Arrow Table
metadata = {"title": args.title} if args.title else {}
table = pa.table(
    {"x": X_swapped[:, 0], "y": X_swapped[:, 1], **cat_cols},
    metadata=metadata
)

# Save arrow file
with pa.OSFile(args.out, "wb") as sink:
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
print("Saving to", args.out)
