"""
buildPyramidConfig.py — generate a pyramid config JSON from obs.tsv.gz

Columns are ranked automatically by biological relevance, cardinality, and null rate.
The top --top_n columns are selected; remainder printed for review.

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
                    help="Number of top-ranked features to include (default: all eligible columns)")
args = parser.parse_args()

obs_path = os.path.join(args.results_dir, args.metadata_file)
print(f"Reading {obs_path} ...")
obs = pd.read_csv(obs_path, sep='\t', low_memory=False)

# Columns to always skip — IDs, QC numerics, ontology term IDs
SKIP_COLS = {
    "Unnamed: 0", "old_index", "barcodes", "sample_id",
    "is_primary_data", "sizeFactor", "scDblFinder_score",
    "prob_compromised", "sum", "detected", "filtered_cell_count",
    "processed_cells", "submitter_id", "cluster",
    "disease_ontology_term_id", "tissue_ontology_term_id",
    "assay_ontology_term_id", "organism_ontology_term_id",
    "organism_ontology_id", "sex_ontology_term_id",
    "development_stage_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
    "harmonized_cell_label_ontology_term_id",
    "singler_celltype_ontology", "scimilarity_celltype_ontology",
    "consensus_celltype_ontology", "collection_doi",
    "scimilarity_min_distance", "consensus_celltype_annotation",
    "n_genes",
}

# High-priority columns — always included regardless of cardinality, null rate, or uniqueness
HIGH_PRIORITY = {
    "harmonized_cell_label",
    "sample_label",
    "author_label",
    "disease",
    "suspension_type",
    "assay",
    "development_stage",
    "tissue",
    "collection_doi_label",   # publication
}

def score_column(col, series):
    """Score a column for viz usefulness. Returns (score, info, feature_type) or (None, reason, None)."""
    n_total = len(series)
    n_null = series.isnull().sum()
    null_rate = n_null / n_total
    n_unique = series.nunique()

    # High priority columns always included — skip null-rate and cardinality checks
    # sample_label gets 110 so it sorts immediately after the pinned harmonized_cell_label
    if col == "sample_label":
        return 110, f"{n_unique} unique, {null_rate:.0%} null", "category"
    elif col in HIGH_PRIORITY:
        return 100, f"{n_unique} unique, {null_rate:.0%} null", "category"

    if null_rate > 0.5:
        return None, f"{null_rate:.0%} null", None
    if n_unique <= 1:
        return None, f"only {n_unique} unique value", None

    is_numeric = np.issubdtype(series.dtype, np.number)
    feature_type = "continuous" if (is_numeric and n_unique > 20) else "category"

    is_cell_label = any(k in col.lower() for k in ["cell_label", "cell_type", "celltype", "annotation", "author_label"])
    if not is_numeric and not is_cell_label and n_unique > 100:
        return None, f"too many categories ({n_unique})", None

    # Infer importance from column name keywords
    col_lower = col.lower()
    keyword_score = 0
    if any(k in col_lower for k in ["cell_type", "celltype", "cell_label", "cell_category"]):
        keyword_score += 40
    elif any(k in col_lower for k in ["harmonized", "annotation"]):
        keyword_score += 30
    elif any(k in col_lower for k in ["disease", "diagnosis", "tumor", "cancer", "cnv"]):
        keyword_score += 28
    elif any(k in col_lower for k in ["tissue", "region", "location"]):
        keyword_score += 25
    elif any(k in col_lower for k in ["sex", "age", "stage", "ethnicity", "race"]):
        keyword_score += 20
    elif any(k in col_lower for k in ["mutation", "subtype", "subdiagnosis", "group"]):
        keyword_score += 18
    elif any(k in col_lower for k in ["status", "event", "survival", "os_", "efs_"]):
        keyword_score += 15
    elif any(k in col_lower for k in ["source", "type", "class", "category"]):
        keyword_score += 12
    elif any(k in col_lower for k in ["pass", "filter", "score", "percent", "mito", "doublet"]):
        keyword_score += 5

    # Data-driven score: cardinality + coverage + entropy
    value_counts = series.dropna().value_counts(normalize=True)
    entropy = -(value_counts * np.log2(value_counts + 1e-10)).sum()
    max_entropy = np.log2(n_unique) if n_unique > 1 else 1
    entropy_score = (entropy / max_entropy) * 20  # 0-20

    cardinality_score = 20 if n_unique <= 20 else 0
    coverage_score = (1 - null_rate) * 10  # 0-10

    base = keyword_score + entropy_score + cardinality_score + coverage_score

    return base, f"{n_unique} unique, {null_rate:.0%} null", feature_type

# Determine the top cell label column to pin first
CELL_LABEL_CANDIDATES = [
    "harmonized_cell_label", "cell_type", "celltype", "cell_label",
    "cell_type_annotation", "author_cell_type",
]
pinned = next((c for c in CELL_LABEL_CANDIDATES if c in obs.columns), None)

# Score all candidate columns
scored = []
rejected = []
for col in obs.columns:
    if col in SKIP_COLS or col == pinned:
        continue
    score, info, feature_type = score_column(col, obs[col])
    if score is not None:
        scored.append((col, score, info, feature_type))
    else:
        rejected.append((col, info))

# Sort by score descending
scored.sort(key=lambda x: -x[1])

# Pin cell label first, then remaining from scored (up to top_n if specified)
top_n_rest = None if args.top_n is None else (args.top_n - 1 if pinned else args.top_n)
selected = []
if pinned:
    selected.append(pinned)
selected += [col for col, _, _, _ in scored[:top_n_rest]]

print(f"\nSelected {len(selected)} features (score | type | info):")
if pinned:
    print(f"  pinned  {pinned}  ({obs[pinned].nunique()} unique, {obs[pinned].isnull().mean():.0%} null)  [category]")
for col, score, info, feature_type in scored[:top_n_rest]:
    print(f"  {score:6.1f}  {col}  ({info})  [{feature_type}]")

remaining = [] if top_n_rest is None else scored[top_n_rest:]
if remaining:
    print(f"\nNot included (edit JSON to add):")
    for col, score, info, feature_type in remaining:
        print(f"  {score:6.1f}  {col}  ({info})  [{feature_type}]")

if rejected:
    print(f"\nRejected {len(rejected)} columns:")
    for col, reason in rejected:
        print(f"  {col}: {reason}")

feature_labels = list(selected)  # copy — edit manually in output JSON
type_map = {col: ft for col, _, _, ft in scored}
if pinned:
    type_map[pinned] = "category"
feature_types = [type_map.get(col, "category") for col in selected]

# Determine output path first so we can check for existing config
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
    print(f"\nFound existing config at {out_path} — reusing reference_name, reference_uuid, target_dir, start_chns")

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
