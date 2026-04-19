# cell_label_harmonization

Pipeline for mapping author cell type labels to Cell Ontology (CL) terms and annotating
single-cell obs metadata with a standardized label hierarchy.

## Pipeline overview

```
obs.tsv.gz
        │
        ├──► detect_hierarchy.py ──► hierarchy.json ──┐
        │                                              │
        └──► generate_remap.py  ──► remap.tsv    ─────┴──► cellLabelMap_obs.py ──► obs_mapped.tsv.gz
                                                                                    obs_mapped_doc.txt
```

## Scripts

### 1. `detect_hierarchy.py`
Reads an obs TSV and infers the parent–child relationships among annotation columns
via functional dependency. Prints the full hierarchy tree and writes a `hierarchy.json`
describing the column structure (parent/child links and selected leaf column).

```bash
python3 detect_hierarchy.py --input obs.tsv.gz
# saves to obs_hierarchy.json in the same directory
# override with --output path/to/hierarchy.json
```

### 2. `generate_remap.py`
Auto-detects the most granular annotation column and maps each unique author label to a
Cell Ontology (CL) term ID using a resolution cascade: exact match → synonym → fuzzy →
LLM fallback. Writes a `remap.tsv` consumed by `cellLabelMap_obs.py`.

```bash
python3 generate_remap.py \
    --input  obs.tsv.gz \
    --obo    cl-basic.obo \
    --output remap.tsv \
    [--column Type_updated] \
    [--pdf paper.pdf] \
    [--doi 10.1234/example]
```

### 3. `cellLabelMap_obs.py`
Combines the remap and hierarchy outputs to produce a fully annotated obs file.
Adds mapped CL term columns (`mapped_cell_label*`) and a three-level label hierarchy
(`level1_cell_label`, `level2_cell_label`, `level3_cell_label`). Either input is optional
— the script runs remap-only or hierarchy-only if only one is provided.

```bash
# Full annotation (remap + hierarchy):
python3 cellLabelMap_obs.py \
    --obs       obs.tsv.gz \
    --remap     remap.tsv \
    --hierarchy hierarchy.json

# Remap only:
python3 cellLabelMap_obs.py --obs obs.tsv.gz --remap remap.tsv

# Hierarchy only:
python3 cellLabelMap_obs.py --obs obs.tsv.gz --hierarchy hierarchy.json
```

## Output columns added to obs

| Column | Source | Description |
|--------|--------|-------------|
| `level3_cell_label` | obs (copy) | Selected leaf annotation column |
| `level2_cell_label` | obs (copy) | Immediate parent of leaf column |
| `level1_cell_label` | obs (copy) | Grandparent of leaf column |
| `mapped_cell_label` | remap | Standardized plain-English cell type name |
| `mapped_cell_label_method` | remap | `cl_term_name`, `glossary`, `llm`, or `original` |
| `mapped_cell_label_ontology_term_id` | remap | Resolved CL term ID |
| `mapped_cell_label_ontology_term_name` | remap | Canonical OBO name for the CL term |
| `mapped_cell_label_ontology_term_match_method` | remap | Resolution method used |

## Supporting module

- **`column_hierarchy.py`** — shared logic for annotation column detection and hierarchy
  inference, imported by both `detect_hierarchy.py` and `generate_remap.py`.

## Tests

- **`test_generate_remap.py`** — unit and integration tests for `generate_remap.py`.
