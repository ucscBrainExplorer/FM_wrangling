#!/usr/bin/env python3
"""
Convert an Arrow IPC file to TSV.

Usage:
  python3 arrow_to_tsv.py --input file.arrow --output file.tsv
"""

import argparse
import pyarrow.ipc as ipc


def build_parser():
    parser = argparse.ArgumentParser(description="Convert Arrow IPC file to TSV")
    parser.add_argument('--input',  required=True, metavar='FILE', help='Input .arrow file')
    parser.add_argument('--output', required=True, metavar='FILE', help='Output .tsv file')
    return parser


def main():
    args = build_parser().parse_args()

    print(f"Reading {args.input}...")
    with ipc.open_file(args.input) as f:
        df = f.read_pandas()
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print(f"Writing {args.output}...")
    df.to_csv(args.output, sep='\t', index=False)
    print("Done.")


if __name__ == '__main__':
    main()
