#!/usr/bin/env python3
"""
Fetch metadata and asset download URL for a single CellxGene dataset by dataset_id.

Usage:
    python3 fetch_cxg_dataset_by_id.py <dataset_id>

Example:
    python3 fetch_cxg_dataset_by_id.py 843c1459-e852-429c-9339-17b5760e7716
"""

import json
import sys
import urllib.request

API_URL = "https://api.cellxgene.cziscience.com/curation/v1/datasets"


def main():
    if len(sys.argv) < 2:
        print("Usage: fetch_cxg_dataset_by_id.py <dataset_id>", file=sys.stderr)
        sys.exit(1)

    target_id = sys.argv[1].strip()

    print(f"Fetching dataset list from CellxGene API...", file=sys.stderr)
    datasets = json.loads(urllib.request.urlopen(API_URL, timeout=60).read())
    print(f"  Total datasets: {len(datasets)}", file=sys.stderr)

    match = next((d for d in datasets if target_id in d.get("dataset_id", "")), None)

    if match is None:
        print(f"ERROR: dataset_id '{target_id}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDataset found:", file=sys.stderr)
    print(f"  dataset_id:         {match.get('dataset_id')}", file=sys.stderr)
    print(f"  dataset_version_id: {match.get('dataset_version_id')}", file=sys.stderr)
    print(f"  collection:         {match.get('collection_name')}", file=sys.stderr)
    print(f"  title:              {match.get('title') or match.get('dataset_title')}", file=sys.stderr)
    print(f"  cells:              {match.get('cell_count')}", file=sys.stderr)

    print(f"\nAssets:", file=sys.stderr)
    for a in match.get("assets", []):
        size_gb = a.get("filesize", 0) / 1e9
        print(f"  {a['filetype']:<6} {size_gb:.2f} GB  {a['url']}", file=sys.stderr)

    print(f"\nFull record (JSON):")
    print(json.dumps(match, indent=2))


if __name__ == "__main__":
    main()
