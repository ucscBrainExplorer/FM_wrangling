"""Which obs column is an atlas's main cell label.

One list for annotate_obs.py (cell_type counts in metadata_summary.json) and
buildPyramidConfig.py (the column pinned first in the pyramid), so the site's
summary and the map use the same labels.
"""

# Highest priority first. Columns added on purpose (harmonized_*, mapped_cell_label) come
# first. concensus_subclass (the HMBA cross-species subclass) is taken
# only when it covers at least CONCENSUS_MIN_COVERAGE (90%) of the cells cell_type covers:
# an HMBA atlas passes (100%, or a little less with a small non-HMBA dataset added), while
# a cohort where only the HMBA datasets have it (adult human atlas: 44%) keeps cell_type.
# subclass is taken only when it covers more cells than cell_type (e.g. only some datasets
# of the cohort have a cell_type); the last three are fallbacks for an obs with no
# cell_type column.
CONCENSUS_MIN_COVERAGE = 0.9

CELL_LABEL_PRIORITY = [
    "harmonized_subclass",   # cross-study subclass labels added to obs (no Cell Ontology terms)
    "harmonized_cell_type",
    "harmonized_cell_label",
    "mapped_cell_label",     # from cell_label_harmonization/cellLabelMap_obs.py
    "concensus_subclass",    # HMBA cross-species subclass (spelling as in the HMBA source)
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
        if col == "concensus_subclass" and "cell_type" in obs.columns \
                and obs["concensus_subclass"].notna().sum() < CONCENSUS_MIN_COVERAGE * obs["cell_type"].notna().sum():
            continue
        return col
    return None
