"""Annotate an atlas's obs.tsv.gz and build the portal's metadata_summary.json.

Run from the directory above FM_wrangling (it reads datasetLists/ and .env by relative path):
  python3 FM_wrangling/annotate_obs.py --data_dir <ATLAS_DIR>

Reads   <ATLAS_DIR>/obs.tsv.gz, <ATLAS_DIR>/dataset_ids.tsv (dataset_id <tab> source h5ad URL),
        datasetLists/{ucsc_datasets.txt, manual_column.json, datasets.info.cxg.<date>.txt}, and the
        manually filled fields (reference_name, reference_uuid, ...) of an existing metadata_summary.json
Writes  obs_custom_annotated.tsv.gz, metadata_summary.json, wrangling_notes_annotation.txt and the two
        system_prompt_*.txt files, into --output_dir (default: --data_dir)
Needs   OPENAI_API_KEY (in .env): organism and age bucket are normalized with gpt-4.1
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Atlas directory holding obs.tsv.gz and dataset_ids.tsv")
parser.add_argument("--output_dir", default=None,
                    help="Where to write the outputs (default: --data_dir); use a scratch dir to compare a re-run")
args = parser.parse_args()

import pandas as pd
import numpy as np
import os, sys, json, re, glob
from typing import Any
import math
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)

# ---------- set up ----------
data_dir = args.data_dir
output_dir = args.output_dir or data_dir
os.makedirs(output_dir, exist_ok=True)
metadata_file = os.path.join(data_dir, "obs.tsv.gz")
metadata_annotated_file = os.path.join(output_dir, "obs_custom_annotated.tsv.gz")
existing_summary_file = os.path.join(data_dir, "metadata_summary.json")  # source of the manually filled fields
metadata_summary_file = os.path.join(output_dir, "metadata_summary.json")
wrangling_notes_file = os.path.join(output_dir, "wrangling_notes_annotation.txt")
wrangling_notes_combineEmbedding_file = os.path.join(data_dir, "wrangling_notes_combineEmbedding.txt") # may not exist
wrangling_notes_custom_file = os.path.join(data_dir, "wrangling_notes_custom.txt") # may not exist
mapped_doc_file = re.sub(r'\.tsv(\.gz)?$', '', metadata_file) + "_doc.txt"  # may not exist
system_prompt_file_normalized_age_bucket = os.path.join(output_dir, "system_prompt_normalized_age_bucket.txt")
system_prompt_file_normalized_organism = os.path.join(output_dir, "system_prompt_normalized_organism.txt")
dataset_uri_file = os.path.join(data_dir, "dataset_ids.tsv")  # dataset_id <tab> source h5ad URL, no header

# info files
ucsc_file = "datasetLists/ucsc_datasets.txt"
manual_column_config_file = "datasetLists/manual_column.json"
# CELLxGENE dataset table: newest datasetLists/datasets.info.cxg.<date>.txt (refresh with
# datasetLists/fetch_cxg_datasets.py); replaces the live census query, which needs AVX
cxg_info_file = sorted(glob.glob("datasetLists/datasets.info.cxg.*.txt"))[-1]

if not os.path.exists(metadata_file):
    sys.exit(f"ERROR: {metadata_file} not found")

fwrangle = open(wrangling_notes_file,'w')

def to_json_safe(obj: Any, **json_kwargs) -> str:
    """
    Serialize Python objects to valid JSON.
    Converts NaN / ±inf to null and enforces JSON compliance.
    """

    def normalize(o):
        if isinstance(o, dict):
            return {k: normalize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [normalize(v) for v in o]
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                return None
        return o

    safe_obj = normalize(obj)
    return json.dumps(safe_obj, allow_nan=False, **json_kwargs)

# ---------- load metadata ----------
meta_data = pd.read_csv(metadata_file, sep ='\t')
print(f"{metadata_file}: {meta_data.shape[0]:,} cells, {meta_data.shape[1]} columns")

# ---------- datasets publication information ----------
# dataset_info: key is "dataset_id"; values are dicts with "collection_doi_label", "collection_doi", "dataset_title"

# dataset info from the cellxgene dataset table (same source as combineEmbedding.py)
print("cellxgene dataset table:", cxg_info_file)
cxg_df = pd.read_csv(cxg_info_file, sep="\t", dtype=str).rename(columns={"#dataset_id": "dataset_id"})
dataset_info_df = cxg_df[["dataset_id", "collection_doi_label", "collection_doi", "dataset_title"]]
dataset_info_df = dataset_info_df.set_index("dataset_id")
dataset_info = dataset_info_df.to_dict(orient="index")
print("cellxgene datasets:", len(dataset_info))

# dataset info from ucsc datasets
ucsc_df = pd.read_csv(ucsc_file, sep="\t", comment = "#")
ucsc_df = ucsc_df[["dataset_id", "collection_doi_label", "collection_doi", "dataset_title"]]
ucsc_df = ucsc_df.set_index("dataset_id")
ucsc = ucsc_df.to_dict(orient="index")
print("ucsc datasets:", len(ucsc))

# combine ucsc with cellxgene 
for idx, vals in ucsc.items():
    if idx in dataset_info:
        print("duplicated datasets detected", idx)
    else:
        dataset_info[idx] = vals
print("all datasets:", len(dataset_info))

# ---------- annotate with publication information, get raw data information ----------
# add columns "collection_doi", "collection_doi_label", "dataset_title";
# publication_summary goes into the metadata summary json
ann_columns = ["collection_doi", "collection_doi_label", "dataset_title"]

note_map = {
    "collection_doi": "publication doi",
    "collection_doi_label": "publication doi label",
    "dataset_title": "dataset title"
}

has_dataset_id = "dataset_id" in meta_data.columns

# annotate with publication information only if dataset_id is available
if has_dataset_id:
    collection_doi_map = {k: v.get("collection_doi") for k, v in dataset_info.items()}
    collection_doi_label_map = {k: v.get("collection_doi_label") for k, v in dataset_info.items()}
    dataset_title_map = {k: v.get("dataset_title") for k, v in dataset_info.items()}
    ann_column_map = {
        "collection_doi": collection_doi_map,
        "collection_doi_label": collection_doi_label_map,
        "dataset_title": dataset_title_map,
    }
    for column in ann_columns:
        meta_data[column] = meta_data["dataset_id"].map(ann_column_map[column])

# Raw data URLs come from dataset_ids.tsv (dataset_id <tab> source h5ad URL, no header): the URL each
# embedding was actually run on. No fallback -- a guessed Census path can name the wrong release or a file
# that does not exist.
if has_dataset_id:
    if not os.path.exists(dataset_uri_file):
        raise FileNotFoundError(f"{dataset_uri_file} not found; list each dataset_id and its source h5ad URL")
    uri_df = pd.read_csv(dataset_uri_file, sep="\t", comment="#", header=None, names=["dataset_id", "uri"], dtype=str)
    dataset_uri_map = dict(zip(uri_df["dataset_id"], uri_df["uri"]))
    print(dataset_uri_map)
    missing = sorted(d for d in meta_data["dataset_id"].unique() if pd.isna(dataset_uri_map.get(d)))
    if missing:
        raise ValueError(f"{dataset_uri_file} has no source URL for: {missing}")
    def build_url(dataset_id):
        return dataset_uri_map[dataset_id]

# collect summary info
groupby_cols = ["collection_doi_label", "collection_doi", "dataset_id"] if has_dataset_id else ["collection_doi_label", "collection_doi", "dataset_title"]
raw_pub_summary = (
    meta_data
    .groupby(groupby_cols, dropna=False)
    .size()
    .reset_index(name="count")
)

if has_dataset_id:
    raw_pub_summary["uri"] = raw_pub_summary["dataset_id"].apply(build_url)
    uri_list = raw_pub_summary["uri"].astype(str).tolist()
    note = "Download raw data from the following locations:"
    fwrangle.write(f"{note}\n")
    for uri in uri_list:
        fwrangle.write(f"{uri}\n")
        fwrangle.flush()

# publication metadata
publication_summary = []

for (label, doi), group in raw_pub_summary.groupby(
    ["collection_doi_label", "collection_doi"], dropna=False
):
    raw_data = []
    for _, row in group.iterrows():
        entry = {"cell_count": int(row["count"])}
        if has_dataset_id:
            entry["dataset_id"] = row["dataset_id"]
            entry["uri"] = row["uri"]
        else:
            entry["dataset_title"] = row["dataset_title"]
        raw_data.append(entry)

    publication_summary.append({
        "label": label,
        "doi": doi,
        "cell_count": int(group["count"].sum()),
        "raw_data": raw_data,
    })

# add wrangling notes
note = "Combine .obs fields from source h5ad files"
fwrangle.write(f"{note}\n")
fwrangle.flush()

for column in ann_columns:
    note = f"Add column '{column}', populate with {note_map[column]} from source data"
    fwrangle.write(f"{note}\n")
    fwrangle.flush()

# ---------- custom information in manual_column.json (custom_map) ----------
# custum column information such as cell_type or filling in missing columns using static values
with open(manual_column_config_file, "r") as f:
    custom_map = json.load(f)

# ---------- annotate with custom_map information ----------
# For each row in meta_data:
#   default: keep meta_data["cell_type"]
#   override for specific dataset_ids: use a different source column (e.g. "supercluster_term"),
#   fill a missing column with a static value (e.g. organism = Homo sapiens), or translate values
if not has_dataset_id:
    print("No dataset_id column — skipping custom_map overrides")
else:
    dataset_ids_data = set(meta_data["dataset_id"].unique())
    for dataset_id_from_map, cfg in custom_map.items():
        if dataset_id_from_map not in dataset_ids_data:
            continue
        mask = (meta_data["dataset_id"] == dataset_id_from_map)
        for metadata_param, param_cfg in cfg.items():
            if not isinstance(param_cfg, dict):
                continue
            
            value = param_cfg.get("value")
            data_type = param_cfg.get("type")

            collection_doi_label = dataset_info[dataset_id_from_map]["collection_doi_label"]
            dataset_title = dataset_info[dataset_id_from_map]["dataset_title"]

            if data_type == "static":
                meta_data.loc[mask, metadata_param] = value
                note = (
                    f"Set '{metadata_param}' = '{value}' for dataset '{dataset_id_from_map}' "
                    f"'{collection_doi_label}' '{dataset_title}'"
                )
                fwrangle.write(f"{note}\n")
                fwrangle.flush()
                print(note)
            elif data_type == "column":
                new_column = metadata_param
                source_col = value
                # combineEmbedding.py applies these rules while building the atlas, then keeps only the
                # selected obs columns, so the source column may be gone; the target is already filled.
                if source_col not in meta_data.columns:
                    skip_note = (
                        f"Skipped mapping '{source_col}' -> '{new_column}' for dataset '{dataset_id_from_map}' "
                        f"'{collection_doi_label}' '{dataset_title}': source column '{source_col}' not in obs"
                    )
                    fwrangle.write(f"{skip_note}\n")
                    fwrangle.flush()
                    print(skip_note)
                    continue
                if new_column not in meta_data.columns:
                    meta_data[new_column] = None
                    default_note = f"Initialized '{new_column}' from '{metadata_param}'"
                    fwrangle.write(f"{default_note}\n")
                    fwrangle.flush()
                    print(default_note)

                meta_data.loc[mask, new_column] = meta_data.loc[mask, source_col]
                custom_note = (
                    f"Map column '{source_col}' -> '{new_column}' "
                    f"for dataset '{dataset_id_from_map}' '{collection_doi_label}' '{dataset_title}'"
                )
                fwrangle.write(f"{custom_note}\n")
                fwrangle.flush()
                print(custom_note)
            elif data_type == "map":
                # Same semantics as combineEmbedding.py: translate source values through param_cfg["map"]
                # (null -> NaN); values not in the map pass through unchanged and are reported.
                new_column = metadata_param
                source_col = value
                value_map = param_cfg.get("map") or {}
                if source_col not in meta_data.columns:
                    skip_note = (
                        f"Skipped value map '{source_col}' -> '{new_column}' for dataset '{dataset_id_from_map}' "
                        f"'{collection_doi_label}' '{dataset_title}': source column '{source_col}' not in obs"
                    )
                    fwrangle.write(f"{skip_note}\n")
                    fwrangle.flush()
                    print(skip_note)
                    continue
                if new_column not in meta_data.columns:
                    meta_data[new_column] = None

                src = meta_data.loc[mask, source_col]
                hit = src.isin(value_map.keys())
                unmapped = sorted(str(v) for v in src[~hit & src.notna()].unique())
                translated = src.map(value_map).where(hit, src)
                meta_data.loc[mask, new_column] = translated.where(translated.notna(), np.nan)  # null -> NaN, not None
                custom_note = (
                    f"Map column '{source_col}' -> '{new_column}' via value map "
                    + ", ".join(f"'{k}'->" + ("NaN" if v is None else f"'{v}'") for k, v in value_map.items())
                    + f" for dataset '{dataset_id_from_map}' '{collection_doi_label}' '{dataset_title}'"
                )
                if param_cfg.get("note"):
                    custom_note += f" -- NOTE: {param_cfg['note']}"
                if unmapped:
                    custom_note += f" -- unmapped values passed through unchanged: {unmapped}"
                fwrangle.write(f"{custom_note}\n")
                fwrangle.flush()
                print(custom_note)

# ---------- normalized organism information (add column "normalized_organism") ----------
system_prompt = """
You are a biological metadata normalization assistant.

Your task is to normalize heterogeneous organism entries into standardized organism names
and assign the corresponding ontology identifier.

You will be given both:
- the ontology term ID ("organism_ontology_term_id")
- the raw label ("organism")

Instructions:
- Use the ontology term ID as the primary source of truth whenever it is available.
- Use the raw label only if the ontology ID is missing, unknown, or cannot be interpreted.
- Normalize each entry to a single accepted scientific name.
- Use standard binomial nomenclature (Genus species).
- Prefer NCBI Taxonomy conventions.
- Do not make assumptions beyond the information provided.

Your raw label may include:
- common names (e.g. "Human", "mouse", "rat")
- scientific names (e.g. "Homo sapiens", "Mus musculus")
- abbreviations (e.g. "H. sapiens")
- strain or subspecies information (e.g. "C57BL/6 mouse", "human iPSC")
- capitalization or spelling variants
- ambiguous or higher-level taxa (e.g. "mammal", "vertebrate")
- unknown or missing values

---

### STANDARDIZED ORGANISM OUTPUT

- Use the **accepted scientific name** (binomial) whenever possible.
- Capitalize genus, lowercase species (e.g. "Homo sapiens").
- Assign the corresponding **NCBI Taxonomy ID** for the normalized organism.
- If the organism cannot be confidently resolved to a species:
  - Use `"unknown"` for both name and ontology ID.
  - Do NOT guess.

---

### Normalization rules (ordered):

Step 0: Unknown (highest priority)
- If the ontology ID or label indicates "unknown", "unspecified", "unclassified",
  or cannot be resolved → assign:
  - normalized organism: "unknown"
  - normalized ontology ID: "unknown"

Step 1: Ontology-driven resolution
- If a valid organism ontology ID is provided:
  - Resolve it to the corresponding accepted scientific name.
  - Assign the matching NCBI Taxonomy ID.
  - Ignore strain, subspecies, or cell line details.

Step 2: Scientific names in label
- If the label already contains a valid binomial scientific name:
  - Normalize capitalization and spacing only.
  - Assign the corresponding NCBI Taxonomy ID.

Step 3: Common names
- Map common names to their accepted scientific name
  (e.g. "human" → "Homo sapiens").
- Assign the corresponding NCBI Taxonomy ID.

Step 4: Abbreviations
- Expand abbreviations where unambiguous
  (e.g. "H. sapiens" → "Homo sapiens").
- Assign the corresponding NCBI Taxonomy ID.

Step 5: Higher-level or ambiguous taxa
- Labels such as "mammal", "vertebrate", "rodent", or other non-species terms
  → assign:
    - normalized organism: "unknown"
    - normalized ontology ID: "unknown"

---

### Output format (JSON ONLY):

Return a JSON object with this structure:

{
  "annotations": [
    {
      "organism_ontology_term_id": "...",
      "organism": "...",
      "normalized_organism": "...",
      "normalized_organism_ontology_term_id": "..."
    }
  ]
}

Each input entry must appear exactly once.  
Do not include explanations or commentary.
"""

# Step 1: compute counts per original label/ontology

org_group_cols = [c for c in ["organism_ontology_term_id", "organism"] if c in meta_data.columns]

organism_counts = (
    meta_data
    .groupby(org_group_cols, dropna=False)
    .size()
    .reset_index(name="count")
    .dropna(subset=org_group_cols, how="all")
)

if "organism_ontology_term_id" not in organism_counts.columns:
    organism_counts["organism_ontology_term_id"] = None

organism_entries = organism_counts.to_dict(orient="records")

# Step 2: send to OpenAI for normalization

user_message = {
    "development_stage_entries": organism_entries
}

client = OpenAI()

print("Sending organism request to OpenAI...", flush=True)
response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_message)}
    ]
)

content = response.output_text
annotations = json.loads(content)["annotations"]

# Convert back to DataFrame
normalized_organism = pd.DataFrame(annotations)

# Drop existing column if present before merge
if "normalized_organism" in meta_data.columns:
    meta_data = meta_data.drop(columns=["normalized_organism"])

# Merge on available columns
merge_cols = [c for c in ["organism_ontology_term_id", "organism"] if c in meta_data.columns]
meta_data = meta_data.merge(
    normalized_organism[merge_cols + ["normalized_organism"]],
    on=merge_cols,
    how="left",
)
note = f"Populate 'normalized_organism' column using OpenAI gpt-4.1 model with 'system_prompt_normalized_organism.txt'"
fwrangle.write(f"{note}\n")
fwrangle.flush()

# QA
df = pd.DataFrame(normalized_organism)
qa_cols = [c for c in ["organism", "organism_ontology_term_id", "normalized_organism"] if c in df.columns]
print(df[qa_cols].to_string(index=False))

# Save system prompt to file
with open(system_prompt_file_normalized_organism, "w") as f:
    f.write(system_prompt)

# ---------- normalized developmental stage information ----------
# normalized_age_bucket_summary and normalized_age_bucket_ontology_term_id_summary for the summary json
# human specific

system_prompt = """
You are a biological metadata normalization assistant.

Your task is to normalize heterogeneous "development_stage" entries into standardized age buckets.

This task applies ONLY to human (Homo sapiens) data.
All normalization and ontology mappings must use human developmental stage concepts (HsapDv).

You will be given both:
- the ontology term ID ("development_stage_ontology_term_id")
- the raw label ("development_stage")

Instructions:
- Use the ontology term ID as the primary source of truth whenever it is available.
- Use the raw label only if the ontology ID is missing, unknown, or cannot be interpreted.
- Normalize each entry into a standardized age bucket as defined below.
- Do not make assumptions beyond the information provided.

Your raw label may include:
- exact ages (e.g. "42-year-old human stage")
- age ranges (e.g. "25–44 year-old human stage")
- decades (e.g. "sixth decade stage")
- developmental terms (e.g. "embryonic human stage", "adult stage")
- prenatal timing (e.g. "16th week post-fertilization", "Carnegie stage 20")
- infancy or childhood descriptions
- unknown or ambiguous labels

---

### STANDARDIZED AGE BUCKETS (ordered):

- prenatal_embryonic             (blastula, Carnegie stages, weeks post-fertilization ≤ 8)
- prenatal_fetal                 (weeks post-fertilization > 8 until birth)
- newborn                        (0-1 month)
- infant                         (1-23 months)
- child                          (2-12 years)
- adolescent                     (13-18 years)
- young_adult                    (19-24 years)
- 25-44_year_old                 (25-44 years)
- middle_aged                    (45-64 years)
- older_adult                    (65-79 years)  
- elderly                        (80+ years)
- adult_unspecified              (labels like "adult", "human adult stage")
- postnatal_unspecified          (postnatal but no age resolution)
- unknown                        (explicitly unknown or cannot be inferred)

---

Use HsapDv whenever possible.  
Only use EFO if no suitable HsapDv term exists.

- prenatal_embryonic  
  → HsapDv:0000002 (embryonic stage)

- prenatal_fetal  
  → HsapDv:0000037 (fetal stage)

- newborn  
  → HsapDv:0000082 (newborn stage)

- infant  
  → HsapDv:0000083 (infant stage)

- child  
  → HsapDv:0000081 (child stage)

- adolescent
  → HsapDv:0000086 (adolescent stage)

- young_adult 
  → HsapDv:0000089 (young adult stage)

- 25-44_year_old 
  → HsapDv:0000090 (25-44 year-old human stage)

- middle_aged  
  → HsapDv:0000092 (45-64 year-old human stage)

- older_adult 
  → HsapDv:0000094 (65-79 year-old stage)

- elderly  
  → HsapDv:0000095 (elderly 80+ year-old)

- adult_unspecified  
  → HsapDv:0000087 (adult stage)

- postnatal_unspecified  
  → unknown

- unknown  
  → unknown

---
### Classification rules:

Step 0: Unknown (highest priority)
- If the ontology ID or label indicates "unknown", "unclassified", or cannot be interpreted → assign "unknown".

Step 1: Prenatal stages
- Carnegie stages → prenatal_embryonic
- Blastula, embryonic → prenatal_embryonic
- Post-fertilization weeks:
    - ≤ 8 weeks → prenatal_embryonic
    - > 8 weeks → prenatal_fetal
- LMP month references → prenatal_fetal

Step 2: Newborn / infant
- "newborn", "0–28 days" → newborn
- months old (<24 months) → infant
- "under-2-year-old" → infant

Step 3: Childhood
- 2–12 years → child
- 13-18 years → adolescent

Step 4: Adults (numeric or decade-based)
- 19–24 → young_adult
- 25-44 → 25-44_year_old 
- 45–64 → middle_aged
- 65–79 → older_adult
- ≥ 80 → elderly
- Decade-based terms (e.g. "sixth decade") should be mapped accordingly.

Step 5: Adult but unspecified
- Labels like "adult stage", "human adult stage", "early adulthood", "late adulthood"
  → adult_unspecified

Step 6: Postnatal but unspecified
- "postnatal stage", "infant stage" (without age)
  → postnatal_unspecified

---

### Output format (JSON ONLY):

Return a JSON object with this structure:

{
  "annotations": [
    {
      "development_stage_ontology_term_id": "...",
      "development_stage": "...",
      "normalized_age_bucket": "..."
      "normalized_age_bucket_ontology_term_id": "..."
    }
  ]
}

Each input entry must appear exactly once.  
Do not include explanations or commentary.

"""

# Step 1:
stage_group_cols = [c for c in ["development_stage_ontology_term_id", "development_stage"]
                    if c in meta_data.columns]

development_stage_counts = (
    meta_data
    .groupby(stage_group_cols, dropna=False)
    .size()
    .reset_index(name="count")
)

if "development_stage_ontology_term_id" not in development_stage_counts.columns:
    development_stage_counts["development_stage_ontology_term_id"] = None

development_stage_entries = development_stage_counts.to_dict(orient="records")

user_message = {
    "development_stage_entries": development_stage_entries
}

client = OpenAI()

print("Sending development stage request to OpenAI...", flush=True)
response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_message)}
    ]
)
print("Response received.", flush=True)

# Extract JSON from the response
content = response.output_text.strip()
# strip markdown fences if present
if content.startswith("```"):
    content = content.split("\n", 1)[-1]
    content = content.rsplit("```", 1)[0].strip()

if not content:
    print("ERROR: empty response from OpenAI. Full response object:")
    print(response)
    sys.exit(1)
annotations = json.loads(content)["annotations"]
# Convert back to DataFrame
normalized_age_bucket = pd.DataFrame(annotations)

# Write note only after successful parse
note = f"Populate 'normalized_age_bucket' using OpenAI gpt-4.1 model with 'system_prompt_normalized_age_bucket.txt'"
fwrangle.write(f"{note}\n")
fwrangle.flush()

# QA
qa_cols = [c for c in ["development_stage", "development_stage_ontology_term_id",
                       "normalized_age_bucket", "normalized_age_bucket_ontology_term_id"]
           if c in normalized_age_bucket.columns]
print(normalized_age_bucket[qa_cols].to_string(index=False))

# Step 2: normalized_age_bucket_summary
# Create a mapping from development_stage -> normalized_age_bucket
stage_to_bucket = dict(
    zip(
        normalized_age_bucket["development_stage"],
        normalized_age_bucket["normalized_age_bucket"]
    )
)

# Map normalized_age_bucket for summary (without modifying meta_data)
mapped_buckets = meta_data["development_stage"].map(stage_to_bucket)

# Count occurrences
counts = mapped_buckets.value_counts().rename_axis("label").reset_index(name="count")

# Apply canonical ordering
AGE_BUCKET_ORDER = [
    "prenatal_embryonic",
    "prenatal_fetal",
    "newborn",
    "infant",
    "child",
    "adolescent",
    "young_adult",
    "25-44_year_old",
    "middle_aged",
    "older_adult",
    "elderly",
    "adult_unspecified",
    "postnatal_unspecified",
    "unknown",
]

counts["label"] = pd.Categorical(counts["label"], categories=AGE_BUCKET_ORDER, ordered=True)
counts = counts.sort_values("label")
    
# Convert to list-of-dicts for JSON summary
normalized_age_bucket_summary = (
    counts
    .dropna(subset=["label"])
    .rename(columns={"count": "cell_count"})
    .to_dict(orient="records")
)

# Step 3: normalized_age_bucket_ontology_term_id_summary
# Create a mapping from development_stage -> normalized_age_bucket_ontology_id
stage_to_bucket = dict(
    zip(
        normalized_age_bucket["development_stage"],
        normalized_age_bucket["normalized_age_bucket_ontology_term_id"]
    )
)

# Map normalized_age_bucket for summary (without modifying meta_data)
mapped_buckets = meta_data["development_stage"].map(stage_to_bucket)

# Count occurrences
counts = mapped_buckets.value_counts().rename_axis("label").reset_index(name="count")

# Convert to list-of-dicts for JSON summary
normalized_age_bucket_ontology_term_id_summary = counts.dropna(subset=["label"]).to_dict(orient="records")

# Save system prompt to file
with open(system_prompt_file_normalized_age_bucket, "w") as f:
    f.write(system_prompt)

# ---------- generate reference data summary json ----------
def getUniqueLabelCounts(meta_data, column):
    counts = meta_data[column].value_counts().to_dict()
    labelCounts = [
        {
            "label": column,
            "cell_count": count,
        }
        for column, count in counts.items()
    ]
    return labelCounts

def safeGetUniqueLabelCounts(meta_data, column):
    if column not in meta_data.columns:
        return []
    return getUniqueLabelCounts(meta_data, column)

existing_summary={}
if os.path.exists(existing_summary_file):
    with open(existing_summary_file, "r") as f:
        existing_summary = json.load(f)

summary = {}
# list of manually filled information
summary["reference_name"] = existing_summary.get("reference_name", "")
summary["reference_uuid"] = existing_summary.get("reference_uuid", "")
summary["version_label"] = existing_summary.get("version_label", "")
summary["reference_pyramid"] = existing_summary.get("reference_pyramid", "")
summary["foundation_model"] = existing_summary.get("foundation_model", "")
summary["abstract"] = existing_summary.get("abstract", "")

# metadata generated from the data
summary["cell_number"] = meta_data.shape[0]

### required metadata used for chart summary
summary["publication"] = publication_summary
summary["tissue"] = getUniqueLabelCounts(meta_data, "tissue")
summary["tissue_type"] = getUniqueLabelCounts(meta_data, "tissue_type")
summary["disease"] = getUniqueLabelCounts(meta_data, "disease")
summary["assay"] = getUniqueLabelCounts(meta_data, "assay")
summary["suspension_type"] = getUniqueLabelCounts(meta_data, "suspension_type")
summary["normalized_age_bucket"] = normalized_age_bucket_summary
summary["organism"] = getUniqueLabelCounts(meta_data, "normalized_organism")
summary["development_stage"] = getUniqueLabelCounts(meta_data, "development_stage")
summary["development_stage_ontology_term_id"] = getUniqueLabelCounts(meta_data, "development_stage_ontology_term_id")

# other parameters
summary["sex"] = safeGetUniqueLabelCounts(meta_data, "sex")
summary["self_reported_ethnicity"] = safeGetUniqueLabelCounts(meta_data, "self_reported_ethnicity")
summary["dataset_ids"] = safeGetUniqueLabelCounts(meta_data, "dataset_id")
summary["tissue_ontology_term_id"] = safeGetUniqueLabelCounts(meta_data, "tissue_ontology_term_id")
summary["disease_ontology_term_id"] = safeGetUniqueLabelCounts(meta_data, "disease_ontology_term_id")
summary["assay_ontology_term_id"] = safeGetUniqueLabelCounts(meta_data, "assay_ontology_term_id")
summary["sex_ontology_term_id"] = safeGetUniqueLabelCounts(meta_data, "sex_ontology_term_id")
summary["self_reported_ethnicity_ontology_term_id"] = safeGetUniqueLabelCounts(meta_data, "self_reported_ethnicity_ontology_term_id")
summary["organism_ontology_term_id"] = safeGetUniqueLabelCounts(meta_data, "organism_ontology_term_id")
summary["normalized_age_bucket_ontology_term_id"] = normalized_age_bucket_ontology_term_id_summary

# cell_type and cell_type_ontology_term_id
# Priority: harmonized_cell_type > harmonized_cell_label > mapped_cell_label > custom_cell_type > cell_type
if "harmonized_cell_type" in meta_data.columns:
    summary["cell_type"] = getUniqueLabelCounts(meta_data, "harmonized_cell_type")
    summary["cell_type_ontology_term_id"] = getUniqueLabelCounts(meta_data, "harmonized_cell_type_ontology_term_id")
    note = "'harmonized_cell_type' is used to generate the cell_type distribution"
    fwrangle.write(f"{note}\n")
    print(note)
elif "harmonized_cell_label" in meta_data.columns:
    summary["cell_type"] = getUniqueLabelCounts(meta_data, "harmonized_cell_label")
    summary["cell_type_ontology_term_id"] = getUniqueLabelCounts(meta_data, "harmonized_cell_label_ontology_term_id")
    note = "'harmonized_cell_label' is used to generate the cell_type distribution"
    fwrangle.write(f"{note}\n")
    print(note)
elif "mapped_cell_label" in meta_data.columns:
    summary["cell_type"] = getUniqueLabelCounts(meta_data, "mapped_cell_label")
    summary["cell_type_ontology_term_id"] = getUniqueLabelCounts(meta_data, "mapped_cell_label_ontology_term_id")
    note = "'mapped_cell_label' is used to generate the cell_type distribution"
    fwrangle.write(f"{note}\n")
    print(note)
elif "custom_cell_type" in meta_data.columns:
    summary["cell_type"] = getUniqueLabelCounts(meta_data, "custom_cell_type")
    summary["cell_type_ontology_term_id"] = getUniqueLabelCounts(meta_data, "custom_cell_type_ontology_term_id")
    note = "'custom_cell_type' is used to generate the cell_type distribution"
    fwrangle.write(f"{note}\n")
    fwrangle.flush()
    print(note)
else:
    summary["cell_type"] = getUniqueLabelCounts(meta_data, "cell_type")
    summary["cell_type_ontology_term_id"] = safeGetUniqueLabelCounts(meta_data, "cell_type_ontology_term_id")

# if there is mapping doc, add at the end
if os.path.exists(mapped_doc_file):
    with open(mapped_doc_file) as f:
        fwrangle.write("\nCell label harmonization performed\n")
        fwrangle.write(f.read())
fwrangle.flush()

# if there is wrangling_notes_combineEmbedding_file, add at the end
if os.path.exists(wrangling_notes_combineEmbedding_file):
    with open(wrangling_notes_combineEmbedding_file) as f:
        fwrangle.write("\nCell label harmonization performed\n")
        fwrangle.write(f.read())
fwrangle.flush()

# if there is wrangling_notes_custom_file, add at the end
if os.path.exists(wrangling_notes_custom_file):
    with open(wrangling_notes_custom_file) as f:
        fwrangle.write("\n")
        fwrangle.write(f.read())
fwrangle.flush()

# output .json file
json_str = to_json_safe(summary, indent=2)
with open(metadata_summary_file, "w") as f:
    f.write(json_str)
print(f"Wrote {metadata_summary_file}")

# ---------- close wrangling notes ----------
fwrangle.close()
print(f"Wrote {wrangling_notes_file}")

# ---------- output obs_custom_annotated.tsv.gz ----------
meta_data.to_csv(metadata_annotated_file, sep="\t", index=False)
print(f"Wrote {metadata_annotated_file}")
