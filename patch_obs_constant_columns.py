"""
patch_obs_constant_columns.py — overwrite one or more columns in an existing
obs.tsv.gz with fixed constant values, in place, WITHOUT changing row order or
row count. For fixing bad derived columns (e.g. collection_doi/dataset_title
pulled from a broken source registry row) on an obs file that's already paired
with a shuffled uce.h5 — reordering isn't an option there, since combineEmbedding.py
generates a fresh random shuffle every run with no seed, and would desync obs from
the already-shuffled embedding.

Reads/writes in chunks (dtype=str throughout) to stay memory-safe on large obs
files, matching the fix applied to combineEmbedding.py/buildPyramidConfig.py.

Usage:
    python3 FM_wrangling/patch_obs_constant_columns.py \\
        --results_dir WB_HMBA_Human_v0.5_shuffle/ \\
        --obs_file obs.tsv.gz \\
        --dataset_id WB_HMBA_Human_AIT_v0.5 \\
        --set collection_doi= \\
        --set "collection_doi_label=Human and Mammalian Brain Atlas v0.5" \\
        --set "dataset_title=WB_HMBA_Human_AIT.h5ad"

--set col=value may be repeated. An empty value (col=) writes empty/NaN.
--dataset_id restricts the patch to rows with that value in the 'dataset_id'
column (omit to patch every row — only safe when the whole file is one dataset).
"""

import argparse
import gzip
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True)
parser.add_argument("--obs_file", default="obs.tsv.gz")
parser.add_argument("--dataset_id", default=None,
                    help="Only patch rows where dataset_id equals this value "
                         "(omit to patch every row in the file)")
parser.add_argument("--set", action="append", required=True, dest="sets",
                    metavar="COL=VALUE",
                    help="Column to overwrite and its new constant value "
                         "(repeatable). COL= (empty value) writes NaN/empty.")
parser.add_argument("--chunksize", type=int, default=500_000)
args = parser.parse_args()

overrides = {}
for spec in args.sets:
    if "=" not in spec:
        raise ValueError(f"--set must be COL=VALUE, got: {spec!r}")
    col, val = spec.split("=", 1)
    overrides[col] = (val if val != "" else np.nan)

obs_path = os.path.join(args.results_dir, args.obs_file)
tmp_path = obs_path + ".patching_tmp"

print(f"Reading {obs_path} in chunks of {args.chunksize:,}, patching {list(overrides)} ...")
if args.dataset_id is not None:
    print(f"  restricted to dataset_id == {args.dataset_id!r}")

header_cols = pd.read_csv(obs_path, sep='\t', nrows=0).columns.tolist()
missing = [c for c in overrides if c not in header_cols]
if missing:
    raise ValueError(f"--set column(s) not found in {obs_path}: {missing}")
if args.dataset_id is not None and "dataset_id" not in header_cols:
    raise ValueError(f"--dataset_id given but obs file has no 'dataset_id' column")

reader = pd.read_csv(obs_path, sep='\t', dtype=str, low_memory=False, chunksize=args.chunksize)

total_rows = 0
patched_rows = 0
header_written = False
with gzip.open(tmp_path, 'wt') as out_f:
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        if args.dataset_id is not None:
            mask = (chunk["dataset_id"] == args.dataset_id).values
        else:
            mask = np.ones(len(chunk), dtype=bool)
        patched_rows += int(mask.sum())
        for col, val in overrides.items():
            chunk.loc[mask, col] = val
        chunk.to_csv(out_f, sep='\t', index=False, header=not header_written)
        header_written = True
        print(f"  chunk {i + 1}: {len(chunk):,} rows ({int(mask.sum()):,} patched), "
              f"cumulative {total_rows:,} rows / {patched_rows:,} patched")

print(f"\nTotal rows read: {total_rows:,}; rows patched: {patched_rows:,}")

# Sanity check before replacing the original: row count must be unchanged.
# compression='gzip' is explicit because tmp_path's name doesn't end in .gz,
# so pandas can't infer it from the extension the way it does for obs_path.
check_count = sum(len(c) for c in pd.read_csv(tmp_path, sep='\t', dtype=str,
                                              compression='gzip', low_memory=False,
                                              chunksize=args.chunksize))
if check_count != total_rows:
    os.remove(tmp_path)
    raise RuntimeError(f"Row count mismatch after patch: read {total_rows:,}, "
                       f"wrote {check_count:,}. Original file left untouched; "
                       f"{tmp_path} removed.")

os.replace(tmp_path, obs_path)
print(f"Verified row count unchanged ({check_count:,}). {obs_path} updated in place.")
