import argparse
import numpy as np
import h5py
from sklearn.decomposition import IncrementalPCA
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True, help="Directory containing uce.h5 and where outputs will be saved")
parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components (default: 50)")
parser.add_argument("--chunk_size", type=int, default=100_000, help="Rows per chunk for incremental PCA (default: 100000)")
parser.add_argument("--group_name", type=str, default="", help="Optional group name suffix for output files")
args = parser.parse_args()

results_dir = args.results_dir
n_components = args.n_components
chunk_size = args.chunk_size
group_name = args.group_name

ipca_outfile = f"ipca_model_{group_name}.pkl" if group_name else "ipca_model.pkl"
pca_outfile  = f"pca_{group_name}.h5"         if group_name else "pca.h5"

uce_path = os.path.join(results_dir, "uce.h5")
print(f"Reading {uce_path}")
fuce = h5py.File(uce_path, "r")
uce  = fuce["data"]
print(f"UCE shape: {uce.shape}")

# Fit
print(f"Fitting IncrementalPCA (n_components={n_components}, chunk_size={chunk_size}) ...")
ipca = IncrementalPCA(n_components=n_components)
total_len = uce.shape[0]
for i in range(0, total_len, chunk_size):
    batch = uce[i:i + chunk_size, :]
    ipca.partial_fit(batch)
    print(f"  fit chunk {i}–{min(i + chunk_size, total_len)}")

joblib.dump(ipca, os.path.join(results_dir, ipca_outfile))
print(f"iPCA model saved to {os.path.join(results_dir, ipca_outfile)}")

# Transform
print("Transforming ...")
n_rows = uce.shape[0]
fpca  = h5py.File(os.path.join(results_dir, pca_outfile), "w")
X_pca = fpca.create_dataset(
    "pca",
    shape=(n_rows, ipca.n_components),
    dtype="float32",
    chunks=(min(chunk_size, n_rows), ipca.n_components),
    compression="gzip"
)
for i in range(0, n_rows, chunk_size):
    batch = uce[i:i + chunk_size, :]
    transformed = ipca.transform(batch)
    X_pca[i:i + len(transformed)] = transformed
    print(f"  transformed rows {i}–{i + len(transformed)}")

fpca.close()
fuce.close()
print(f"PCA saved to {os.path.join(results_dir, pca_outfile)}  shape: ({n_rows}, {ipca.n_components})")
