#!/usr/bin/env python3
"""
Convert a 2-column TSV file (x, y) to Apache Arrow format.

Example usage:
    python tsv_to_arrow.py input.tsv output.arrow
"""

import argparse
import pyarrow as pa
import pyarrow.ipc as ipc
import csv
import sys


def tsv_to_arrow(input_path: str, output_path: str, has_header: bool = True) -> int:
    """Convert a TSV file with 2 numeric columns into an Arrow file.
    Returns the number of rows written.
    Raises exceptions on failure.
    """
    with open(input_path, newline='') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        rows = list(tsv_reader)

    if has_header:
        rows = rows[1:]  # drop header

    # convert to float, assuming two columns per row
    try:
        data = [[float(row[0]), float(row[1])] for row in rows]
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing data from {input_path}: {e}")

    # create Arrow arrays
    x = pa.array([d[0] for d in data], pa.float32())
    y = pa.array([d[1] for d in data], pa.float32())

    # create Arrow table
    table = pa.table([x, y], names=["x", "y"])

    # write to Arrow file
    with pa.OSFile(output_path, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    return len(data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 2-column TSV file to Arrow (.arrow) format."
    )
    parser.add_argument("input", help="Path to input TSV file")
    parser.add_argument("output", help="Path to output Arrow file")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Indicate that the TSV file does not have a header row",
    )

    args = parser.parse_args()

    try:
        num_rows = tsv_to_arrow(args.input, args.output, has_header=not args.no_header)
        print(f"✅ Successfully wrote {num_rows} rows to {args.output}")
    except FileNotFoundError:
        sys.stderr.write(f"Error: File not found - {args.input}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
