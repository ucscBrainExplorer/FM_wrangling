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
parser.add_argument("--pinned", default=None,
                    help="Column to pin as the primary cell label (overrides CELL_LABEL_CANDIDATES)")
parser.add_argument("--columns", default=None,
                    help="Comma-separated list of columns to restrict the build to "
                         "(for a partial/first-run config). All other columns are ignored "
                         "regardless of HIGH_PRIORITY/eligibility. --pinned label is kept "
                         "even if omitted from this list.")
args = parser.parse_args()

obs_path = os.path.join(args.results_dir, args.metadata_file)

# Validate --pinned/--columns before paying the cost of loading the full file
header_cols = None
if args.pinned or args.columns:
    header_cols = pd.read_csv(obs_path, sep='\t', nrows=0).columns.tolist()

if args.pinned and args.pinned not in header_cols:
    raise ValueError(f"--pinned column '{args.pinned}' not found in {obs_path}\n"
                     f"Available columns: {header_cols}")

restrict_cols = None
if args.columns:
    restrict_cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    missing = [c for c in restrict_cols if c not in header_cols]
    if missing:
        raise ValueError(f"--columns entries not found in {obs_path}: {missing}\n"
                         f"Available columns: {header_cols}")

# Candidate columns for auto-pinning the primary cell label, if --pinned isn't given
CELL_LABEL_CANDIDATES = [
    "harmonized_cell_label", "custom_cell_type", "cell_type", "celltype",
    "cell_type_annotation", "author_cell_type",
]

# When --columns restricts the build, only read those columns off disk (plus the
# pinned label / label candidates, so auto-pinning still works) instead of the
# full file — a 70-column obs file can otherwise blow up memory for no reason.
usecols = None
if restrict_cols is not None:
    keep = set(restrict_cols)
    if args.pinned:
        keep.add(args.pinned)
    else:
        keep.update(c for c in CELL_LABEL_CANDIDATES if c in header_cols)
    usecols = [c for c in header_cols if c in keep]

print(f"Reading {obs_path} ...")
if usecols is not None:
    print(f"  (restricted to {len(usecols)} columns: {usecols})")

# low_memory=False buffers the whole column set in memory to sniff dtypes, which
# is wasteful on a large obs file — usecols alone doesn't avoid it, since dtype
# sniffing happens before columns are dropped. Instead sniff dtypes from a cheap
# sample, then do a single-pass typed read of just the needed columns (or all of
# them, if --columns wasn't given). Widen sampled ints to float64 in case a NaN
# shows up outside the sample (int dtype can't hold NaN).
sample = pd.read_csv(obs_path, sep='\t', usecols=usecols, nrows=200_000)
dtype_map = {
    c: ("float64" if pd.api.types.is_integer_dtype(sample[c]) else sample[c].dtype)
    for c in sample.columns
}
del sample
obs = pd.read_csv(obs_path, sep='\t', usecols=usecols, dtype=dtype_map)

# Columns to always skip — IDs, QC numerics, ontology term IDs
SKIP_COLS = {
    "Unnamed: 0", "old_index", "barcodes",
    "is_primary_data", "sizeFactor",
    "prob_compromised", "sum", "detected", "filtered_cell_count",
    "processed_cells", "submitter_id",
    "disease_ontology_term_id", "tissue_ontology_term_id",
    "assay_ontology_term_id", "organism_ontology_term_id",
    "organism_ontology_id", "sex_ontology_term_id",
    "development_stage_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
    "harmonized_cell_label_ontology_term_id",
    "harmonized_cell_label_ontology_term_match_method",
    "singler_celltype_ontology", "scimilarity_celltype_ontology",
    "consensus_celltype_ontology", "collection_doi",
    "n_genes",
    "x", "y",
    "x_ccf", "y_ccf", "z_ccf",
    "x_section", "y_section", "z_section",
}

# High-priority columns — always included, listed before alphabetical remainder
# sample_label is ordered first within this group (after the pinned cell label)
HIGH_PRIORITY = {
    "sample_label",
    "harmonized_cell_label",
    "author_label",
    "disease",
    "suspension_type",
    "assay",
    "development_stage",
    "tissue",
    "tissue_type",
    "collection_doi_label",
    "publication",
    "normalized_organism",
    "subclass",
    "class",
}

FORCE_CATEGORY = {"cluster"}

# When obs spans multiple collections/datasets with disjoint metadata schemas
# (e.g. two studies concatenated into one obs table, each contributing its own
# annotation columns), a column that's fully populated within its own study
# still looks globally "mostly null" once the other study's cells — which
# never had that field — are counted too. Check null rate per collection
# instead of globally, so we don't drop one study's real metadata just
# because a larger co-mingled study doesn't carry that field at all.
GROUP_COL = next((c for c in ("dataset_id", "collection_doi_label") if c in obs.columns), None)
MIN_GROUP_SIZE = max(50, int(0.01 * len(obs)))  # ignore tiny/noise groups

group_sizes = None
if GROUP_COL is not None:
    group_sizes = obs[GROUP_COL].value_counts()
    if (group_sizes >= MIN_GROUP_SIZE).sum() > 1:
        print(f"Detected {(group_sizes >= MIN_GROUP_SIZE).sum()} collections via "
              f"'{GROUP_COL}' — using per-collection null-rate check")
    else:
        GROUP_COL = None  # only one real group, no benefit to grouping

def classify_column(col, series):
    """Returns (feature_type, reject_reason) — reject_reason is None if eligible."""
    n_total = len(series)
    null_rate = series.isnull().sum() / n_total

    if GROUP_COL is not None and col != GROUP_COL:
        # Best-case null rate among sizeable collections — a column fully
        # populated within its own collection shouldn't be rejected just
        # because other collections don't carry that field at all.
        valid_groups = group_sizes[group_sizes >= MIN_GROUP_SIZE].index
        per_group_null = series.groupby(obs[GROUP_COL]).apply(lambda s: s.isnull().mean())
        per_group_null = per_group_null.reindex(valid_groups).dropna()
        if len(per_group_null):
            null_rate = min(null_rate, per_group_null.min())

    n_unique = series.nunique()
    is_numeric = np.issubdtype(series.dtype, np.number)
    feature_type = "continuous" if (is_numeric and n_unique > 20 and col.lower() not in FORCE_CATEGORY) else "category"

    if col in HIGH_PRIORITY or (restrict_cols is not None and col in restrict_cols):
        return feature_type, None

    if null_rate > 0.5:
        global_null_rate = series.isnull().sum() / n_total
        return None, f"{global_null_rate:.0%} null"

    is_cell_label = any(k in col.lower() for k in
                        ["cell_type", "celltype", "annotation", "author_label"])
    if not is_numeric and not is_cell_label and n_unique > n_total / 10:
        return None, f"too many categories ({n_unique})"

    return feature_type, None

# Determine the top cell label column to pin first (CELL_LABEL_CANDIDATES defined above)
if args.pinned:
    pinned = args.pinned
else:
    pinned = next((c for c in CELL_LABEL_CANDIDATES if c in obs.columns), None)

# Classify all candidate columns
eligible = {}   # col -> feature_type
rejected = []

candidate_cols = obs.columns if restrict_cols is None else restrict_cols
for col in candidate_cols:
    if col in SKIP_COLS or col == pinned or col.endswith("_term_id"):
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
selected += high_priority_cols + rest_cols

# If normalized_organism is present, drop organism (redundant)
if "normalized_organism" in selected:
    selected = [c for c in selected if c != "organism"]

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
    "feature_labels": ["organism" if c == "normalized_organism" else c for c in selected],
}

with open(out_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"\nConfig written to {out_path}")
print("Fill in: reference_name, reference_uuid, target_dir, start_chns")
