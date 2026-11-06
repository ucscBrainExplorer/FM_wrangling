"""The CELLxGENE dataset table, with citation overrides applied.

One loader for combineEmbedding.py and annotate_obs.py, so both use the same
snapshot (newest datasetLists/datasets.info.cxg.<date>.txt, from
datasetLists/fetch_cxg_datasets.py) and the same corrected citations
(datasetLists/doi_overrides.txt).
"""
import glob
import os
import sys

import pandas as pd


def load_cxg_info(lists_dir):
    """(path of the snapshot used, its table with dataset_id as a column)."""
    files = sorted(glob.glob(os.path.join(lists_dir, "datasets.info.cxg.*.txt")))
    if not files:
        sys.exit(f"ERROR: no datasets.info.cxg.*.txt in {lists_dir}; run {lists_dir}/fetch_cxg_datasets.py")
    path = files[-1]
    df = pd.read_csv(path, sep="\t", dtype=str).rename(columns={"#dataset_id": "dataset_id"})

    overrides_file = os.path.join(lists_dir, "doi_overrides.txt")
    if os.path.exists(overrides_file):
        ov = pd.read_csv(overrides_file, sep="\t", comment="#", dtype=str)
        for _, row in ov.iterrows():
            hit = df["dataset_id"] == row["dataset_id"]
            if not hit.any():
                print(f"  doi_overrides.txt: {row['dataset_id']} not in {os.path.basename(path)}; skipped")
                continue
            df.loc[hit, ["collection_doi", "collection_doi_label"]] = [row["collection_doi"], row["collection_doi_label"]]
        print(f"  doi_overrides.txt: {len(ov)} citation override(s)")
    return path, df
