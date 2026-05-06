import sys
import os
import argparse
import math

parser = argparse.ArgumentParser()
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--npy", help="UCE embeddings .npy file")
input_group.add_argument("--h5", help="UCE embeddings HDF5 file (dataset key: 'data')")
input_group.add_argument("--h5ad", help="UCE adata .h5ad file")
parser.add_argument("--obs", metavar="FILE", help="obs TSV (required with --npy/--h5; optional override with --h5ad)")
parser.add_argument("--config", required=True, metavar="JSON", help="transform config, e.g. Wang2025Nature/transform_e100.json")
parser.add_argument("--out", default="newdata.arrow", help="output Arrow IPC file (default: newdata.arrow)")
parser.add_argument("--title", help="title stored in Arrow schema metadata")
parser.add_argument("--n-bins", type=int, default=10, help="bins for continuous features (default: 10)")
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

sys.path.insert(0, "/mnt/efsUCE/umap_pytorch")
import umap_pytorch
print(umap_pytorch.__file__)

# ── column selection (mirrors buildPyramidConfig.py) ─────────────────────────

SKIP_COLS = {
    "Unnamed: 0", "old_index", "barcodes", "sample_id",
    "is_primary_data", "sizeFactor",
    "prob_compromised", "sum", "detected", "filtered_cell_count",
    "processed_cells", "submitter_id",
    "disease_ontology_term_id", "tissue_ontology_term_id",
    "assay_ontology_term_id", "organism_ontology_term_id",
    "organism_ontology_id", "sex_ontology_term_id",
    "development_stage_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
    "harmonized_cell_label_ontology_term_id",
    "singler_celltype_ontology", "scimilarity_celltype_ontology",
    "consensus_celltype_ontology", "collection_doi",
    "consensus_celltype_annotation",
    "n_genes",
}


CELL_LABEL_CANDIDATES = [
    "harmonized_cell_label", "cell_type", "celltype", "cell_label",
    "cell_type_annotation", "author_cell_type",
]

FORCE_CATEGORY = {"cluster"}

def classify_column(col, series):
    """Returns (feature_type, reject_reason) — reject_reason is None if eligible."""
    n_total = len(series)
    null_rate = series.isnull().sum() / n_total
    n_unique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)
    feature_type = "continuous" if (is_numeric and n_unique > 20 and col.lower() not in FORCE_CATEGORY) else "category"

    if null_rate > 0.5:
        return None, f"{null_rate:.0%} null"
    if n_unique <= 1:
        return None, f"only {n_unique} unique value"

    is_cell_label = any(k in col.lower() for k in
                        ["label", "celltype", "cell_type", "annotation", "harmoniz", "region"])
    if not is_numeric and not is_cell_label and n_unique > 100:
        return None, f"too many categories ({n_unique})"

    return feature_type, None

def bin_series(series, n_bins=10):
    """Bin a numeric series using nice round edges, trimmed to actual data range.
    Returns (str_series, ordered_labels) where labels sort correctly alphabetically."""
    values = series.to_numpy(dtype=float)
    mask = ~np.isnan(values)
    valid = values[mask]

    vmin, vmax = valid.min(), valid.max()
    step_raw = (vmax - vmin) / n_bins
    magnitude = 10 ** math.floor(math.log10(step_raw))
    nice_step = next(m * magnitude for m in [1, 2, 2.5, 5, 10] if m * magnitude >= step_raw)
    nice_min = math.floor(vmin / nice_step) * nice_step
    nice_max = math.ceil(vmax / nice_step) * nice_step
    actual_n_bins = round((nice_max - nice_min) / nice_step)
    bins = np.array([nice_min + i * nice_step for i in range(actual_n_bins + 1)])

    if nice_step >= 1:
        fmt = ".0f"
    else:
        n_dec = max(1, -int(math.floor(math.log10(nice_step))))
        fmt = f".{n_dec}f"

    all_labels = [f"{bins[i]:{fmt}}–{bins[i+1]:{fmt}}" for i in range(actual_n_bins)]

    bin_indices = np.digitize(valid, bins)
    bin_indices = np.clip(bin_indices, 1, actual_n_bins)

    # Remove empty bins so the legend never has zero-count entries
    bin_counts = np.bincount(bin_indices, minlength=actual_n_bins + 1)[1:]
    non_empty = np.where(bin_counts > 0)[0]
    labels = [all_labels[i] for i in non_empty]
    old_to_new = {old + 1: new + 1 for new, old in enumerate(non_empty)}
    bin_indices = np.vectorize(old_to_new.get)(bin_indices)

    result = pd.array([None] * len(values), dtype=pd.StringDtype())
    result[mask] = [labels[i - 1] for i in bin_indices]
    str_series = pd.Series(result, index=series.index).fillna("nan")

    return str_series, labels

# ── load data ─────────────────────────────────────────────────────────────────

with open(args.config, "r") as f:
    config = json.load(f)

config_dir = os.path.dirname(os.path.abspath(args.config))
pca_model_file = os.path.join(config_dir, config["pca_model_file"])
pumap_model_file = os.path.join(config_dir, config["pumap_model_file"])
swap_xy_matrix = np.array(config["swap_xy_matrix"])
d_hat_file = config.get("d_hat")
W = np.load(os.path.join(config_dir, d_hat_file)).astype(np.float32) if d_hat_file else None

ipca = joblib.load(pca_model_file)
pumap = umap_pytorch.load_pumap(pumap_model_file)

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
    obs = pd.read_csv(args.obs, sep="\t", index_col=0) if args.obs else adata.obs

# ── project onto reference UMAP ───────────────────────────────────────────────

X_pca = ipca.transform(uce).astype(np.float32)
if W is not None:
    X_pca = X_pca - (X_pca @ W.T) @ W
data_tensor = torch.from_numpy(X_pca)
X_umap = pumap.transform(data_tensor)
X_swapped = X_umap @ swap_xy_matrix

# ── build column dict (same ordering as buildPyramidConfig.py) ────────────────

pinned = next((c for c in CELL_LABEL_CANDIDATES if c in obs.columns), None)

eligible = {}   # col -> feature_type
for col in obs.columns:
    if col in SKIP_COLS or col == pinned:
        continue
    feature_type, reason = classify_column(col, obs[col])
    if feature_type is not None:
        eligible[col] = feature_type
    else:
        print(f"  skip {col}: {reason}")

ordered_cols = []
if "sample_label" in eligible:
    ordered_cols.append("sample_label")
if pinned:
    ordered_cols.append(pinned)
ordered_cols += sorted([c for c in eligible if c != "sample_label"])

cat_cols = {}
for col in ordered_cols:
    series = obs[col] if col != pinned else obs[col]
    feature_type = "category" if col == pinned else eligible.get(col, "category")

    if feature_type == "continuous":
        str_series, labels = bin_series(series, n_bins=args.n_bins)
        cat_cols[col] = pa.array(pd.Categorical(str_series, categories=labels, ordered=True))
        print(f"  continuous {col}: {len(labels)} bins")
    else:
        str_series = series.astype(str)
        cat_cols[col] = pa.array(str_series.astype("category"))
        print(f"  category   {col}: {series.nunique()} unique")

# ── write arrow file ──────────────────────────────────────────────────────────

metadata = {"title": args.title} if args.title else {}
table = pa.table(
    {"x": X_swapped[:, 0], "y": X_swapped[:, 1], **cat_cols},
    metadata=metadata
)

with pa.OSFile(args.out, "wb") as sink:
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)

print(f"Saved {len(table)} rows → {args.out}")
