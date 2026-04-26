#!/usr/bin/env python3
"""
cellLabelMap_obs.py — annotate an obs TSV with remap results and hierarchy labels.

The source column (join key) is read from the remap file header line
"# Source column: <col>".

The remap file requires four columns:
  original_label
  mapped_cell_label_ontology_term_id
  mapped_cell_label_ontology_term_name
  mapped_cell_label_ontology_term_match_method

After joining, ontology columns are renamed to <source_col>_ontology_term_*.

The hierarchy JSON (from detect_hierarchy.py) provides the two parent columns
above the cell type column, added as level1_cell_label and level2_cell_label.

Added columns
-------------
  <source_col>_ontology_term_id           (remap only)
  <source_col>_ontology_term_name         (remap only)
  <source_col>_ontology_term_match_method (remap only)
  level3_cell_label                       copy of selected cell type col (always added)
  level2_cell_label                       immediate parent of cell type col
  level1_cell_label                       grandparent of cell type col

Output files (derived from --obs)
----------------------------------
  <stem>_mapped.tsv.gz
  <stem>_mapped_doc.txt

Usage
-----
  # Both (full annotation):
  python3 cellLabelMap_obs.py --obs obs.tsv.gz --remap map.tsv --hierarchy hierarchy.json

  # Remap only (<source_col>_ontology_term_* columns, no level1/level2):
  python3 cellLabelMap_obs.py --obs obs.tsv.gz --remap map.tsv

  # Hierarchy only (level1/level2 columns, no ontology terms):
  python3 cellLabelMap_obs.py --obs obs.tsv.gz --hierarchy hierarchy.json
"""

import argparse
import json
import re
import sys
from datetime import datetime

import pandas as pd


REMAP_COLS = [
    "original_label",
    "mapped_cell_label_ontology_term_id",
    "mapped_cell_label_ontology_term_name",
    "mapped_cell_label_ontology_term_match_method",
]


def read_source_column(remap_path):
    """Parse '# Source column: <col>' from the remap file header."""
    with open(remap_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# Source column:"):
                return line.split(":", 1)[1].strip()
    return None


def get_level_cols(hierarchy, source_col):
    """Return (level2_col, level1_col) by walking up the parent chain."""
    columns = hierarchy["columns"]
    level2_col = columns.get(source_col, {}).get("parent")
    level1_col = columns.get(level2_col, {}).get("parent") if level2_col else None
    return level2_col, level1_col


def output_paths(obs_path):
    """Derive output paths by stripping .tsv or .tsv.gz and appending _mapped.*"""
    stem = re.sub(r'\.tsv(\.gz)?$', '', obs_path)
    return stem + "_mapped.tsv.gz", stem + "_mapped_doc.txt"


def write_doc(doc_path, args, output_path, source_col, level2_col, level1_col,
              n_total, n_mapped, unmapped, obs):
    lines = []
    lines.append("cellLabelMap_obs.py — annotation documentation")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("Input files")
    lines.append(f"  obs:       {args.obs}")
    lines.append(f"  remap:     {args.remap or '—'}")
    lines.append(f"  hierarchy: {args.hierarchy or '—'}")
    lines.append(f"  output:    {output_path}")
    lines.append("")

    if args.hierarchy:
        lines.append("Cell type hierarchy")
        lines.append(f"  level1_cell_label  ← obs column '{level1_col}'" if level1_col
                     else "  level1_cell_label  — not available")
        lines.append(f"  level2_cell_label  ← obs column '{level2_col}'" if level2_col
                     else "  level2_cell_label  — not available")
        lines.append(f"  level3_cell_label  ← obs column '{source_col}'")
        lines.append(f"  {source_col}_ontology_term_*  ← remap of obs column '{source_col}'" if args.remap
                     else f"  {source_col}_ontology_term_*  — not available (no remap)")
        lines.append("")

    if args.remap:
        lines.append("Mapping coverage")
        lines.append(f"  Total cells:  {n_total:,}")
        lines.append(f"  Mapped cells: {n_mapped:,} ({100 * n_mapped / n_total:.1f}%)")
        lines.append(f"  Unmapped:     {n_total - n_mapped:,}")
        lines.append("")

        if not unmapped.empty:
            lines.append(f"Unmapped labels ({len(unmapped)} unique)")
            for label, count in unmapped.items():
                lines.append(f"  {count:>6,}  {label}")
            lines.append("")

        lines.append("Mapping method breakdown (unique labels)")
        match_col = f"{source_col}_ontology_term_match_method"
        method_counts = (
            obs[[match_col, source_col]]
            .drop_duplicates(subset=source_col)
            [match_col]
            .value_counts()
        )
        for method, count in method_counts.items():
            lines.append(f"  {count:>4}  {method}")
        lines.append("")

    if args.hierarchy:
        lines.append("Level 1 categories")
        if level1_col and "level1_cell_label" in obs.columns:
            for val in sorted(obs["level1_cell_label"].dropna().unique()):
                lines.append(f"  {val}")
        else:
            lines.append("  —")
        lines.append("")

        lines.append("Level 2 categories")
        if level2_col and "level2_cell_label" in obs.columns:
            for val in sorted(obs["level2_cell_label"].dropna().unique()):
                lines.append(f"  {val}")
        else:
            lines.append("  —")
        lines.append("")

    if args.hierarchy:
        lines.append("Level 3 categories")
        if "level3_cell_label" in obs.columns:
            for val in sorted(obs["level3_cell_label"].dropna().unique()):
                lines.append(f"  {val}")
        else:
            lines.append("  —")
        lines.append("")

    if args.remap:
        id_col   = f"{source_col}_ontology_term_id"
        name_col = f"{source_col}_ontology_term_name"
        meth_col = f"{source_col}_ontology_term_match_method"
        lines.append(f"Remap ({source_col} → ontology term)")
        lines.append(f"  {args.remap}")
        lines.append("")
        label_map = (
            obs[[source_col, id_col, name_col, meth_col]]
            .drop_duplicates()
            .sort_values(source_col)
        )
        for _, row in label_map.iterrows():
            term_id  = row[id_col]  if pd.notna(row[id_col])  else "—"
            term_name = row[name_col] if pd.notna(row[name_col]) else ""
            method   = row[meth_col] if pd.notna(row[meth_col]) else ""
            lines.append(f"  {row[source_col]:<55}  {term_id:<12}  {term_name:<50}  {method}")
        lines.append("")

    with open(doc_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate obs metadata with remap results and hierarchy labels."
    )
    parser.add_argument("--obs", required=True,
                        help="Input obs TSV file (may be .gz compressed)")
    parser.add_argument("--remap",
                        help="Remap TSV from generate_remap.py")
    parser.add_argument("--hierarchy",
                        help="Hierarchy JSON from detect_hierarchy.py")
    args = parser.parse_args()

    if not args.remap and not args.hierarchy:
        parser.error("at least one of --remap or --hierarchy is required")

    output_path, doc_path = output_paths(args.obs)

    # --- source column and hierarchy ---
    hierarchy = None
    level2_col = level1_col = None

    if args.hierarchy:
        with open(args.hierarchy) as f:
            hierarchy = json.load(f)
        source_col = hierarchy["selected"]
        level2_col, level1_col = get_level_cols(hierarchy, source_col)
    else:
        source_col = read_source_column(args.remap)
        if source_col is None:
            print("Error: '# Source column:' not found in remap file header.", file=sys.stderr)
            sys.exit(1)

    print(f"Cell type column (level3): '{source_col}'")
    if level2_col is not None:
        print(f"Level 2 column: '{level2_col}'")
    if level1_col is not None:
        print(f"Level 1 column: '{level1_col}'")

    # --- obs ---
    print(f"Reading {args.obs} ...", end=" ", flush=True)
    obs = pd.read_csv(args.obs, sep="\t", dtype=str, low_memory=False)
    print(f"{len(obs):,} cells, {len(obs.columns)} columns")

    if source_col not in obs.columns:
        print(f"Error: column '{source_col}' not found in obs.", file=sys.stderr)
        print(f"Available columns: {list(obs.columns)}", file=sys.stderr)
        sys.exit(1)

    # --- remap ---
    if args.remap:
        remap = pd.read_csv(args.remap, sep="\t", comment="#", dtype=str)
        missing = [c for c in REMAP_COLS if c not in remap.columns]
        if missing:
            print(f"Error: remap file missing columns: {missing}", file=sys.stderr)
            sys.exit(1)
        remap = remap[REMAP_COLS].drop_duplicates(subset="original_label")
        remap = remap.rename(columns={
            "original_label": source_col,
            "mapped_cell_label_ontology_term_id":           f"{source_col}_ontology_term_id",
            "mapped_cell_label_ontology_term_name":         f"{source_col}_ontology_term_name",
            "mapped_cell_label_ontology_term_match_method": f"{source_col}_ontology_term_match_method",
        })
        obs = obs.merge(remap, on=source_col, how="left")

    # --- add level1 / level2 / level3 from existing obs columns ---
    if args.hierarchy and source_col in obs.columns:
        obs["level3_cell_label"] = obs[source_col]

    if level2_col:
        if level2_col in obs.columns:
            obs["level2_cell_label"] = obs[level2_col]
        else:
            print(f"Warning: level2 column '{level2_col}' not found in obs — skipping.")

    if level1_col:
        if level1_col in obs.columns:
            obs["level1_cell_label"] = obs[level1_col]
        else:
            print(f"Warning: level1 column '{level1_col}' not found in obs — skipping.")

    # --- coverage ---
    n_total = len(obs)
    if args.remap:
        id_col = f"{source_col}_ontology_term_id"
        n_mapped = obs[id_col].notna().sum()
        print(f"Coverage: {n_mapped:,} / {n_total:,} cells ({100 * n_mapped / n_total:.1f}%)")
        unmapped = (
            obs.loc[obs[id_col].isna(), source_col]
            .value_counts()
        )
        if not unmapped.empty:
            print(f"Unmapped labels ({len(unmapped)} unique):")
            for label, count in unmapped.items():
                print(f"  {count:>6,}  {label}")
    else:
        n_mapped = 0
        unmapped = pd.Series(dtype=str)

    # --- output ---
    obs.to_csv(output_path, sep="\t", index=False)
    print(f"Written: {output_path}")

    write_doc(doc_path, args, output_path, source_col, level2_col, level1_col,
              n_total, n_mapped, unmapped, obs)
    print(f"Written: {doc_path}")


if __name__ == "__main__":
    main()
