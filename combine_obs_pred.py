#!/usr/bin/env python3
"""
Combine a UCE kNN prediction TSV with an obs metadata TSV on cell barcode.

Usage:
    python combine_obs_pred.py <pred_tsv> <obs_tsv> [output_tsv]

The pred_tsv may contain leading comment lines starting with '#', which are
skipped. The first column of each file is treated as the cell barcode index.
Prediction columns come first, obs columns are appended on the right.

If output_tsv is omitted, the result is written next to pred_tsv with
'_combined' appended before the extension.
"""

import sys
import argparse
import pandas as pd
from io import StringIO
from pathlib import Path


def load_pred(path: str) -> pd.DataFrame:
    """Load prediction TSV, skipping leading '#' comment lines."""
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return pd.read_csv(StringIO("".join(lines)), sep="\t", index_col=0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pred_tsv", help="Prediction TSV (may have leading # comments)")
    parser.add_argument("obs_tsv", help="Obs metadata TSV (first col = cell barcode)")
    parser.add_argument(
        "output_tsv",
        nargs="?",
        help="Output path (default: pred_tsv stem + '_combined.tsv')",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred_tsv)
    out_path = Path(args.output_tsv) if args.output_tsv else \
        pred_path.with_name(pred_path.stem + "_combined.tsv")

    print(f"Loading pred: {args.pred_tsv}")
    pred = load_pred(args.pred_tsv)
    print(f"  {pred.shape[0]:,} cells x {pred.shape[1]} columns")

    print(f"Loading obs:  {args.obs_tsv}")
    obs = pd.read_csv(args.obs_tsv, sep="\t", index_col=0)
    print(f"  {obs.shape[0]:,} cells x {obs.shape[1]} columns")

    assert pred.shape[0] == obs.shape[0], (
        f"Row count mismatch: pred has {pred.shape[0]:,} cells, obs has {obs.shape[0]:,} cells"
    )
    assert pred.index.equals(obs.index), (
        f"Cell barcodes differ between files — indices are not identical (same count but different or reordered barcodes)"
    )

    combined = pred.join(obs, how="left")
    print(f"Output: {combined.shape[0]:,} cells x {combined.shape[1]} columns -> {out_path}")

    combined.to_csv(out_path, sep="\t")
    print("Done.")


if __name__ == "__main__":
    main()
