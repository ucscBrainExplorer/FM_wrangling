#!/usr/bin/env python3
"""
Fetch CellxGENE dataset metadata and write two files:

  1. datasets.info.cxg.YYYY-MM-DD.txt  — main per-dataset TSV
     Columns: #dataset_id  dataset_total_cell_count  collection_doi  collection_doi_label  dataset_title  is_primary_data  organism

  2. datasets.stats.cxg.YYYY-MM-DD.txt — summary statistics (not per-dataset)
     Sections: total counts, is_primary_data breakdown, organism breakdown

Usage:
  python3 fetch_cxg_datasets.py [options]

Options:
  --out-dir DIR       Directory for output files (default: the datasetLists/ folder holding this script)
  --organism LABEL    Keep only datasets matching this organism label.
                      Common values: "Homo sapiens", "Mus musculus"
                      Can be specified multiple times. Default: no filter.
  --primary-only      Keep only datasets where is_primary_data contains True.
  --include-tombstone Include tombstoned (retracted) datasets (default: excluded).
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from datetime import date

API_URL = "https://api.cellxgene.cziscience.com/curation/v1/datasets"


def fetch_datasets():
    req = urllib.request.Request(API_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def organism_labels(dataset):
    orgs = dataset.get("organism", [])
    if isinstance(orgs, list):
        return [o["label"] if isinstance(o, dict) else str(o) for o in orgs]
    return [str(orgs)]


def primary_data_label(dataset):
    ipd = dataset.get("is_primary_data", [])
    if isinstance(ipd, list):
        vals = set(ipd)
    else:
        vals = {bool(ipd)}
    if vals == {True}:
        return "True"
    elif vals == {False}:
        return "False"
    else:
        return "Mixed"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)), help="Output directory")
    parser.add_argument(
        "--organism", action="append", default=[], metavar="LABEL",
        help="Filter by organism label (repeatable)"
    )
    parser.add_argument(
        "--primary-only", action="store_true",
        help="Keep only is_primary_data=True datasets"
    )
    parser.add_argument(
        "--include-tombstone", action="store_true",
        help="Include tombstoned (retracted) datasets"
    )
    args = parser.parse_args()

    today = date.today().isoformat()
    main_path  = f"{args.out_dir}/datasets.info.cxg.{today}.txt"
    stats_path = f"{args.out_dir}/datasets.stats.cxg.{today}.txt"

    print("Fetching dataset list from CellxGENE API...", file=sys.stderr)
    datasets = fetch_datasets()
    print(f"  Total from API: {len(datasets)}", file=sys.stderr)

    skipped_tombstone = skipped_organism = skipped_primary = 0
    kept = []

    # Summary counters (over kept datasets)
    primary_counter  = Counter()
    organism_counter = Counter()

    for d in datasets:
        if not args.include_tombstone and d.get("tombstone"):
            skipped_tombstone += 1
            continue

        if args.primary_only and primary_data_label(d) not in ("True", "Mixed"):
            skipped_primary += 1
            continue

        if args.organism:
            orgs = organism_labels(d)
            if not any(f in orgs for f in args.organism):
                skipped_organism += 1
                continue

        kept.append(d)
        primary_counter[primary_data_label(d)] += 1
        for org in organism_labels(d):
            organism_counter[org] += 1

    # Write main info file (per-dataset)
    with open(main_path, "w") as fh:
        fh.write("#dataset_id\tdataset_total_cell_count\tcollection_doi\tcollection_doi_label\tdataset_title\tis_primary_data\torganism\n")
        for d in kept:
            orgs = "|".join(organism_labels(d))
            fh.write(
                f"{d['dataset_id']}\t"
                f"{d.get('cell_count') or d.get('dataset_total_cell_count') or 0}\t"
                f"{d.get('collection_doi') or ''}\t"
                f"{d.get('collection_doi_label') or ''}\t"
                f"{d.get('title') or ''}\t"
                f"{primary_data_label(d)}\t"
                f"{orgs}\n"
            )

    # Write summary stats file
    total_cells = sum(d.get('cell_count') or d.get('dataset_total_cell_count') or 0 for d in kept)
    with open(stats_path, "w") as fh:
        fh.write(f"# CellxGENE dataset summary — {today}\n\n")

        fh.write("[totals]\n")
        fh.write(f"datasets\t{len(kept)}\n")
        fh.write(f"total_cells\t{total_cells}\n\n")

        fh.write("[is_primary_data]\n")
        for label, cnt in sorted(primary_counter.items()):
            fh.write(f"{label}\t{cnt}\n")
        fh.write("\n")

        fh.write("[organism]\n")
        for org, cnt in organism_counter.most_common():
            fh.write(f"{org}\t{cnt}\n")

    # Print summary
    print(f"\n=== Summary ({today}) ===", file=sys.stderr)
    print(f"Datasets written: {len(kept)}", file=sys.stderr)
    if skipped_tombstone:
        print(f"  Skipped (tombstone):    {skipped_tombstone}", file=sys.stderr)
    if skipped_primary:
        print(f"  Skipped (non-primary):  {skipped_primary}", file=sys.stderr)
    if skipped_organism:
        print(f"  Skipped (organism):     {skipped_organism}", file=sys.stderr)

    print("\nis_primary_data breakdown:", file=sys.stderr)
    for label, cnt in sorted(primary_counter.items()):
        print(f"  {label:<8} {cnt}", file=sys.stderr)

    print("\nOrganism breakdown:", file=sys.stderr)
    for org, cnt in organism_counter.most_common():
        print(f"  {org:<30} {cnt}", file=sys.stderr)

    print(f"\nMain file:  {main_path}", file=sys.stderr)
    print(f"Stats file: {stats_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
