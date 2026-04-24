"""
buildPyramidConfig.py — generate a pyramid config JSON from obs.tsv.gz

Columns are ordered: pinned cell label, HIGH_PRIORITY features, then
all remaining eligible columns alphabetically.

Usage:
    python3 FM_wrangling/buildPyramidConfig.py \
        --results_dir scpca_glioma_shuffled/ \
        --umap_file umap_e20.npy

Edit the output JSON to fill in:
    - reference_name
    - reference_uuid
    - target_dir
    - start_chns
    - feature_labels (human-readable overrides)
    - add/remove features as needed
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True,
                    help="Directory containing umap .npy and obs file")
parser.add_argument("--umap_file", required=True,
                    help="UMAP npy filename inside results_dir (e.g. umap_e20.npy)")
parser.add_argument("--metadata_file", default="obs.tsv.gz",
                    help="Obs metadata filename inside results_dir (default: obs.tsv.gz)")
parser.add_argument("--out", default=None,
                    help="Output JSON path (default: <results_dir>/pyramid_e{N}_config.json)")
parser.add_argument("--top_n", type=int, default=None,
                    help="Number of top features to include (default: all eligible columns)")
args = parser.parse_args()

obs_path = os.path.join(args.results_dir, args.metadata_file)
print(f"Reading {obs_path} ...")
obs = pd.read_csv(obs_path, sep='\t', low_memory=False)

# Columns to always skip — IDs, QC numerics, ontology term IDs
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

# High-priority columns — always included, listed before alphabetical remainder
# sample_label is ordered first within this group (after the pinned cell label)
HIGH_PRIORITY = {
    "harmonized_cell_label",
    "sample_label",
    "author_label",
    "disease",
    "suspension_type",
    "assay",
    "development_stage",
    "tissue",
    "collection_doi_label",
}

def classify_column(col, series):
    """Returns (feature_type, reject_reason) — reject_reason is None if eligible."""
    n_total = len(series)
    null_rate = series.isnull().sum() / n_total
    n_unique = series.nunique()
    is_numeric = np.issubdtype(series.dtype, np.number)
    feature_type = "continuous" if (is_numeric and n_unique > 20) else "category"

    if col in HIGH_PRIORITY:
        return feature_type, None

    if null_rate > 0.5:
        return None, f"{null_rate:.0%} null"
    if n_unique <= 1:
        return None, f"only {n_unique} unique value"

    is_cell_label = any(k in col.lower() for k in
                        ["cell_label", "cell_type", "celltype", "annotation", "author_label"])
    if not is_numeric and not is_cell_label and n_unique > 100:
        return None, f"too many categories ({n_unique})"

    return feature_type, None

# Determine the top cell label column to pin first
CELL_LABEL_CANDIDATES = [
    "harmonized_cell_label", "cell_type", "celltype", "cell_label",
    "cell_type_annotation", "author_cell_type",
]
pinned = next((c for c in CELL_LABEL_CANDIDATES if c in obs.columns), None)

# Classify all candidate columns
eligible = {}   # col -> feature_type
rejected = []

for col in obs.columns:
    if col in SKIP_COLS or col == pinned:
        continue
    feature_type, reason = classify_column(col, obs[col])
    if feature_type is not None:
        eligible[col] = feature_type
    else:
        rejected.append((col, reason))

# Build selection: pinned cell label → sample_label → other HIGH_PRIORITY (alpha) → rest (alpha)
high_priority_cols = sorted(
    [c for c in eligible if c in HIGH_PRIORITY and c != "sample_label"]
)
rest_cols = sorted([c for c in eligible if c not in HIGH_PRIORITY])

selected = []
if pinned:
    selected.append(pinned)
if "sample_label" in eligible:
    selected.append("sample_label")
selected += high_priority_cols + rest_cols

# Apply top_n if specified
if args.top_n is not None:
    selected = selected[:args.top_n]

# Print summary
type_map = {pinned: "category"} if pinned else {}
type_map.update(eligible)

print(f"\nSelected {len(selected)} features:")
for col in selected:
    series = obs[col]
    n_unique = series.nunique()
    null_rate = series.isnull().mean()
    marker = " [HIGH_PRIORITY]" if col in HIGH_PRIORITY or col == pinned else ""
    print(f"  {col}  ({n_unique} unique, {null_rate:.0%} null)  [{type_map.get(col, 'category')}]{marker}")

if rejected:
    print(f"\nRejected {len(rejected)} columns:")
    for col, reason in rejected:
        print(f"  {col}: {reason}")

feature_types = [type_map.get(col, "category") for col in selected]

# Determine output path
if args.out:
    out_path = args.out
else:
    import re
    m = re.search(r'e(\d+)', args.umap_file)
    epoch_str = f"_e{m.group(1)}" if m else ""
    out_path = os.path.join(args.results_dir, f"pyramid{epoch_str}_config.json")

# Read existing config to carry over reference_name, reference_uuid, target_dir, start_chns
existing = {}
if os.path.exists(out_path):
    with open(out_path) as f:
        existing = json.load(f)
    print(f"\nFound existing config at {out_path} — reusing reference_name, reference_uuid, target_dir")

config = {
    "reference_name": existing.get("reference_name", ""),
    "reference_uuid": existing.get("reference_uuid", ""),
    "target_dir":     existing.get("target_dir", ""),
    "umap_file": args.umap_file,
    "metadata_file": args.metadata_file,
    "features": selected,
    "feature_types": feature_types,
    "n_bins": 10,
    "start_chns": list(range(len(selected))),
    "feature_labels": list(selected),
}

with open(out_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"\nConfig written to {out_path}")
print("Fill in: reference_name, reference_uuid, target_dir, start_chns")
