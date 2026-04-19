#!/usr/bin/env python3
"""
detect_hierarchy.py — show the cell type annotation column hierarchy in a TSV/CSV.

Auto-detects annotation columns, infers parent-child relationships via
functional dependency, and prints the full hierarchy tree with all categorical
values at each level.

Usage
-----
  python3 detect_hierarchy.py --input obs_annotated.tsv[.gz]
  python3 detect_hierarchy.py --input obs_annotated.csv --sep ','
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from column_hierarchy import _REVISED_PATTERN, _cell_type_score, detect_full_hierarchy


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _sorted_values(series):
    return sorted(set(str(v) for v in series if pd.notna(v)))


def _print_subtree(col, children_of, df, nunique, prefix, is_last):
    connector = "└── " if is_last else "├── "
    tag = "  [LEAF]" if not children_of[col] else ""
    print(f"{prefix}{connector}{col}  ({nunique[col]} categories){tag}")

    child_prefix = prefix + ("    " if is_last else "│   ")

    for v in _sorted_values(df[col]):
        print(f"{child_prefix}  · {v}")

    kids = sorted(children_of[col], key=lambda c: nunique[c])
    for i, child in enumerate(kids):
        _print_subtree(child, children_of, df, nunique,
                       prefix=child_prefix, is_last=(i == len(kids) - 1))


def build_output(df, candidates, parent_of, children_of,
                 roots, leaves, nunique, selected):
    """Build a JSON-serialisable dict representing the selected hierarchy chain only.

    Walks from the selected leaf up through parent_of to collect only the
    columns that form the ancestry path of the selected column.
    """
    # Collect the ancestry chain from leaf up to root
    chain = []
    col = selected
    while col is not None:
        chain.append(col)
        col = parent_of.get(col)
    chain.reverse()  # root → leaf order

    return {
        "cell_type_col": selected,
        "columns": {
            col: {
                "n_categories": nunique[col],
                "parent": parent_of[col],
                "values": _sorted_values(df[col]),
            }
            for col in chain
        },
    }


def print_hierarchy(df, candidates, parent_of, children_of,
                    roots, leaves, nunique, cellxgene_cols):

    print(f"\nDataset: {len(df):,} cells  ·  {len(df.columns)} columns total")
    print(f"Annotation columns detected: {len(candidates)}")

    if cellxgene_cols:
        print(f"Skipped cellxgene label columns: {sorted(cellxgene_cols)}")

    if not candidates:
        print("No annotation columns found.")
        return

    # ---- tree ----
    print(f"\n{'─' * 62}")
    print("Column hierarchy  (coarse → fine)")
    print(f"{'─' * 62}")

    sorted_roots = sorted(roots, key=lambda c: candidates.index(c))
    for i, root in enumerate(sorted_roots):
        _print_subtree(root, children_of, df, nunique,
                       prefix="", is_last=(i == len(sorted_roots) - 1))

    # ---- selected leaf ----
    if leaves:
        selected = max(leaves, key=lambda c: (
            bool(_REVISED_PATTERN.search(c)),
            _cell_type_score(df[c].dropna().unique().tolist()) >= 0.3,
            nunique[c],
            candidates.index(c)
        ))
        print(f"\n{'─' * 62}")
        print(f"cell_type_col (most granular): '{selected}'  ({nunique[selected]} categories)")
    else:
        selected = max(candidates, key=lambda c: (nunique[c], candidates.index(c)))
        print(f"\n{'─' * 62}")
        print(f"cell_type_col (most granular fallback): '{selected}'  ({nunique[selected]} categories)")

    # ---- linear chain ----
    chain = []
    col = selected
    while col is not None:
        chain.append(col)
        col = parent_of.get(col)
    chain.reverse()
    print(f"\n{'─' * 62}")
    print("Hierarchy chain:")
    print("  " + " → ".join(chain))

    # ---- summary table ----
    print(f"\n{'─' * 62}")
    print(f"  {'Column':<38} {'#categories':>12}  {'ct_score':>8}  role")
    print(f"{'─' * 62}")
    for col in candidates:
        parts = []
        if col in roots:
            parts.append("root")
        if not children_of[col]:
            parts.append("leaf")
        if col == selected:
            parts.append("cell_type_col")
        role = ", ".join(parts) if parts else "intermediate"
        score = _cell_type_score(df[col].dropna().unique().tolist())
        print(f"  {col:<38} {nunique[col]:>12}  {score:>8.2f}  {role}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect cell type annotation hierarchy in a TSV/CSV file."
    )
    parser.add_argument("--input", required=True,
                        help="Input TSV or CSV file (may be .gz compressed)")
    parser.add_argument("--sep", default="\t",
                        help="Column separator (default: tab). Use ',' for CSV.")
    parser.add_argument("--output", default=None,
                        help="Path to save hierarchy JSON (default: <input_stem>_hierarchy.json next to input).")
    args = parser.parse_args()

    if args.output is None:
        stem = re.sub(r'\.tsv(\.gz)?$|\.csv(\.gz)?$', '', Path(args.input).name)
        args.output = str(Path(args.input).parent / f"{stem}_hierarchy.json")

    print(f"Reading {args.input} ...", end=" ", flush=True)
    try:
        df = pd.read_csv(args.input, sep=args.sep, dtype=str, low_memory=False)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(df):,} rows × {len(df.columns):,} columns")

    candidates, parent_of, children_of, roots, leaves, nunique, cellxgene_cols = \
        detect_full_hierarchy(df)

    print_hierarchy(df, candidates, parent_of, children_of,
                    roots, leaves, nunique, cellxgene_cols)

    if args.output:
        if leaves:
            selected = max(leaves, key=lambda c: (
                bool(_REVISED_PATTERN.search(c)),
                nunique[c],
                candidates.index(c)
            ))
        else:
            selected = max(candidates, key=lambda c: (nunique[c], candidates.index(c)))
        data = build_output(df, candidates, parent_of, children_of,
                            roots, leaves, nunique, selected)
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nHierarchy saved to {args.output}")


if __name__ == "__main__":
    main()
