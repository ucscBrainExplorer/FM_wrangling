import sys
import json
import joblib
import torch
import numpy as np
import pandas as pd
import anndata
import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, "/mnt/efsUCE/umap_pytorch")  #git+https://github.com/ucscBrainExplorer/umap_pytorch.git
import umap_pytorch
print(umap_pytorch.__file__)

uce_adata_file = "tmpdata/e4ddac12-f48f-4455-8e8d-c2a48a683437-logNorm1M_uce_adata.h5ad"
config_path = "Wang2025Nature/transform_e100.json"
arrow_file = "newdata.arrow"

# Read JSON file
with open(config_path, "r") as f:
    config = json.load(f)

# Access fields
pca_model_file = config["pca_model_file"]
pumap_model_file = config["pumap_model_file"]
swap_xy_matrix = np.array(config["swap_xy_matrix"])  # convert to NumPy array

# read pca model back, pumap model back
ipca = joblib.load(pca_model_file)
pumap = umap_pytorch.load_pumap(pumap_model_file)

# read uce result back
adata = anndata.read_h5ad(uce_adata_file, backed='r') # use backed mode to skip loading .X gene expression matrix, save time and memory
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

# Convert to Arrow Table
cell_type = pa.array(obs["cell_type"].astype("category"))
table = pa.table({
    "x": X_swapped[:, 0],
    "y": X_swapped[:, 1],
    "cell_type":cell_type
    })

# Save arrow file
with pa.OSFile(arrow_file, "wb") as sink:
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
print("Saving to", arrow_file)
