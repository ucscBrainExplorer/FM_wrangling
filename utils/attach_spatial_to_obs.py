"""
Join one or more spatial TSVs onto an obs TSV, adding spatial columns in place.
When multiple --spatial files are given, cell IDs must be unique across all of them.
Rows with no spatial data get NaN.

Usage:
    python FM_wrangling/utils/attach_spatial_to_obs.py \
        --obs     Zhuang-ABCA-1_cxg_shuffled/obs.tsv.gz \
        --spatial UCE_npy_obs/73c39ba5_spatial.tsv \
        --out     Zhuang-ABCA-1_cxg_shuffled/obs_spatial.tsv.gz

    python FM_wrangling/utils/attach_spatial_to_obs.py \
        --obs     Zhuang-ABCA-1-2_cxg_shuffled/obs.tsv.gz \
        --spatial UCE_npy_obs/73c39ba5_spatial.tsv UCE_npy_obs/836fc2fc_spatial.tsv \
        --out     Zhuang-ABCA-1-2_cxg_shuffled/obs_spatial.tsv.gz
"""
import argparse
import sys
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--obs",        required=True)
parser.add_argument("--spatial",    required=True, nargs="+")
parser.add_argument("--out",        required=True)
parser.add_argument("--obs_id_col", default=None,
                    help="Column in obs to use as join key instead of the index (e.g. old_index)")
args = parser.parse_args()

print(f"Loading obs: {args.obs}")
obs = pd.read_csv(args.obs, sep="\t", index_col=0)
print(f"  {len(obs):,} rows")

if args.obs_id_col:
    print(f"  using '{args.obs_id_col}' as join key")
    obs = obs.set_index(args.obs_id_col)

parts = []
for path in args.spatial:
    print(f"Loading spatial: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")
    parts.append(df)

if len(parts) > 1:
    all_ids = [idx for df in parts for idx in df.index]
    if len(all_ids) != len(set(all_ids)):
        dupes = len(all_ids) - len(set(all_ids))
        sys.exit(f"ERROR: {dupes:,} duplicate cell IDs across spatial files — aborting.")
    print(f"  cell IDs unique across {len(parts)} files ✓")
    spatial = pd.concat(parts)
else:
    spatial = parts[0]

existing = [c for c in spatial.columns if c in obs.columns]
if existing:
    print(f"  overwriting existing columns: {existing}")
    obs = obs.drop(columns=existing)

obs = obs.join(spatial, how="left")

n = len(obs)
for col in spatial.columns:
    missing = obs[col].isna().sum()
    print(f"  missing {col}: {missing:,} / {n:,} ({100*missing/n:.2f}%)")

print(f"Writing: {args.out}")
obs.to_csv(args.out, sep="\t", compression="gzip" if args.out.endswith(".gz") else None)
print("Done.")
