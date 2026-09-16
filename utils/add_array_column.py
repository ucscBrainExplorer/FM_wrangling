from __future__ import annotations

"""
Append a 1D array (e.g. cluster labels from run_clustering.py) as a new
column on an obs/prediction TSV.

Two merge modes, chosen automatically:
  - Barcode-verified merge: if --array is a .tsv/.csv with a 'cell_id' column
    (e.g. run_clustering.py run with --obs) AND --obs also has a 'cell_id'
    column, cells are matched by barcode identity, not row position. Set
    equality of cell_id is checked, and the merge uses validate='one_to_one'
    (same pattern as FoundationModelBenchmarking/cell_labelling/
    join_predictions_with_obs.py) so a duplicate or mismatched barcode is a
    hard error, not a silent row-count coincidence.
  - Positional merge (fallback): if either side lacks 'cell_id' (e.g. --array
    is a plain .npy, or an older single-column clusters.tsv without
    barcodes), rows are matched by order only, after checking row counts
    match. It's on you to confirm the array and the TSV came from the same
    cell ordering in this case.

For .tsv/.csv without a 'cell_id' column, the new obs column is named after
that file's own header unless --column-name overrides it; a plain .npy has
no header at all, so --column-name is required for it.

Usage:
    python add_array_column.py --obs cell_typing/ruchi_2_WB_HMBA_Human_V0.5.pred_with_obs.tsv \
        --array clusters.tsv --column-name leiden_cluster \
        --output cell_typing/ruchi_2_WB_HMBA_Human_V0.5.pred_with_obs_clusters.tsv
"""

import argparse
import os

import numpy as np
import pandas as pd


def load_array(path: str) -> tuple[np.ndarray | None, np.ndarray, str | None]:
    """Returns (cell_ids_or_None, values, source_column_name); source_column_name is None for .npy."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tsv", ".csv"):
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
        if "cell_id" in df.columns:
            value_cols = [c for c in df.columns if c != "cell_id"]
            if len(value_cols) != 1:
                raise ValueError(f"Expected exactly one non-cell_id column in {path}, found {value_cols}")
            return df["cell_id"].to_numpy(), df[value_cols[0]].to_numpy(), value_cols[0]
        return None, df.iloc[:, 0].to_numpy(), df.columns[0]
    return None, np.load(path), None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs", required=True, help="Path to the obs/prediction TSV")
    parser.add_argument("--array", required=True, help="Path to a 1D (n_cells,) .npy array or single-column .tsv/.csv (optionally with a 'cell_id' column)")
    parser.add_argument(
        "--column-name",
        default=None,
        help="Name for the new column. Defaults to the source file's own (non-cell_id) header for .tsv/.csv; required for .npy",
    )
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    obs = pd.read_csv(args.obs, sep="\t")
    print(f"Loaded obs: {obs.shape[0]:,} cells x {obs.shape[1]} columns from {args.obs}")

    cell_ids, arr, source_column_name = load_array(args.array)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape {arr.shape} from {args.array}")
    print(f"Loaded array: {arr.shape[0]:,} values from {args.array}")

    column_name = args.column_name or source_column_name
    if column_name is None:
        raise ValueError("--column-name is required when --array is a .npy file (no header to default to)")

    if cell_ids is not None and "cell_id" in obs.columns:
        print("Both --array and --obs have cell_id -- doing a barcode-verified merge")
        if set(cell_ids) != set(obs["cell_id"]):
            raise ValueError("cell_id sets differ between --array and --obs -- these are not the same cells")

        array_df = pd.DataFrame({"cell_id": cell_ids, column_name: arr})
        if column_name in obs.columns:
            print(f"Warning: overwriting existing column '{column_name}'")
            obs = obs.drop(columns=[column_name])

        obs = obs.merge(array_df, on="cell_id", how="inner", validate="one_to_one")
        if len(obs) != len(cell_ids):
            raise ValueError(f"Merge produced {len(obs):,} rows, expected {len(cell_ids):,} -- unexpected duplicate/dropped barcodes")
    else:
        if cell_ids is not None and "cell_id" not in obs.columns:
            print("Warning: --array has cell_id but --obs doesn't -- falling back to unverified positional merge")
        if len(arr) != len(obs):
            raise ValueError(
                f"Row count mismatch: obs has {len(obs):,} cells, array has {len(arr):,} values -- "
                "these must come from the same cell ordering"
            )
        if column_name in obs.columns:
            print(f"Warning: overwriting existing column '{column_name}'")
        obs[column_name] = arr

    obs.to_csv(args.output, sep="\t", index=False)
    print(f"Saved {obs.shape[0]:,} cells x {obs.shape[1]} columns -> {args.output}")
    print(f"  {column_name} value counts:")
    print(obs[column_name].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
