#!/usr/bin/env python3
"""
Download Ensembl gene ID → gene symbol mapping tables from BioMart for each
UCE-supported species. Outputs one TSV per species with columns:
  ensembl_gene_id  gene_name

Usage:
    python download_ensembl_maps.py
    python download_ensembl_maps.py --output-dir /path/to/output
    python download_ensembl_maps.py --list-datasets   # verify dataset names

Then upload to S3:
    aws s3 sync model_files/ensembl_maps s3://your-bucket/uce_model_files/ensembl_maps/
"""

import argparse
import sys
import time
from pathlib import Path

import requests

BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

# BioMart dataset name follows the pattern {first_letter_genus}{species}_gene_ensembl.
# Maps UCE species name → BioMart dataset name.
SPECIES_DATASETS = {
    'human':               'hsapiens_gene_ensembl',
    'mouse':               'mmusculus_gene_ensembl',
    'zebrafish':           'drerio_gene_ensembl',
    'frog':                'xtropicalis_gene_ensembl',
    'mouse_lemur':         'mmurinus_gene_ensembl',
    'pig':                 'sscrofa_gene_ensembl',
    'macaca_fascicularis': 'mfascicularis_gene_ensembl',
    'macaca_mulatta':      'mmulatta_gene_ensembl',
}

BIOMART_QUERY = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1"
       uniqueRows="1" datasetConfigVersion="0.6">
  <Dataset name="{dataset}" interface="default">
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="external_gene_name"/>
  </Dataset>
</Query>"""


def list_datasets():
    """Print all available gene datasets from Ensembl BioMart."""
    resp = requests.get(BIOMART_URL,
                        params={"type": "datasets", "mart": "ENSEMBL_MART_ENSEMBL"},
                        timeout=60)
    resp.raise_for_status()
    print("Available datasets (name | display name):")
    for line in resp.text.splitlines():
        fields = line.split("\t")
        if len(fields) >= 3 and "gene_ensembl" in fields[1]:
            print(f"  {fields[1]:<45} {fields[2]}")


def download_mapping(species, dataset, output_dir, retries=3):
    out_path = output_dir / f"{species}_ensembl_map.tsv"
    if out_path.exists():
        print(f"  {species}: already exists, skipping ({out_path})")
        return

    query = BIOMART_QUERY.format(dataset=dataset)
    for attempt in range(1, retries + 1):
        try:
            print(f"  {species} ({dataset})... ", end="", flush=True)
            resp = requests.post(BIOMART_URL, data={"query": query}, timeout=120)
            resp.raise_for_status()
            if resp.text.startswith("Query ERROR"):
                raise ValueError(f"BioMart error: {resp.text.strip()}")
            break
        except Exception as e:
            if attempt == retries:
                print(f"FAILED after {retries} attempts: {e}")
                return
            print(f"attempt {attempt} failed ({e}), retrying...")
            time.sleep(5)

    lines = resp.text.splitlines()
    # Filter out rows with empty gene name (second column)
    rows = [l for l in lines[1:] if l.strip() and l.split("\t")[1].strip()]
    out_path.write_text("\n".join([lines[0]] + rows) + "\n")
    print(f"{len(rows)} mappings -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Ensembl→Hugo gene symbol mapping tables from BioMart."
    )
    parser.add_argument("--output-dir", default="./model_files/ensembl_maps",
                        help="Directory to write TSV files (default: ./model_files/ensembl_maps)")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List available BioMart datasets and exit (useful for verifying names)")
    parser.add_argument("--species", nargs="+", choices=SPECIES_DATASETS,
                        help="Download only these species (default: all)")
    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    species_to_run = args.species or list(SPECIES_DATASETS)
    print(f"Downloading {len(species_to_run)} mapping table(s) to {output_dir}/\n")
    for species in species_to_run:
        download_mapping(species, SPECIES_DATASETS[species], output_dir)

    print("\nDone.")
    print(f"\nUpload to S3:")
    print(f"  aws s3 sync {output_dir} s3://<bucket>/uce_model_files/ensembl_maps/")


if __name__ == "__main__":
    main()
