"""Which obs column is an atlas's main cell label.

One list for annotate_obs.py (cell_type counts in metadata_summary.json) and
buildPyramidConfig.py (the column pinned first in the pyramid), so the site's
summary and the map use the same labels.
"""

# Highest priority first. subclass is taken only when it covers more cells than
# cell_type (e.g. only some datasets of the cohort have a cell_type); the last
# three are fallbacks for an obs with no cell_type column.
CELL_LABEL_PRIORITY = [
    "harmonized_subclass",   # cross-study subclass labels added to obs (no Cell Ontology terms)
    "harmonized_cell_type",
    "harmonized_cell_label",
    "mapped_cell_label",     # from cell_label_harmonization/cellLabelMap_obs.py
    "custom_cell_type",
    "subclass",
    "cell_type",
    "celltype", "cell_type_annotation", "author_cell_type",
]


def pick_cell_label(obs):
    """The main cell label column of DataFrame `obs`, or None if it has none."""
    for col in CELL_LABEL_PRIORITY:
        if col not in obs.columns:
            continue
        if col == "subclass" and "cell_type" in obs.columns \
                and obs["subclass"].notna().sum() <= obs["cell_type"].notna().sum():
            continue
        return col
    return None
