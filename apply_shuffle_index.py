"""Re-apply a saved shuffle to a regenerated obs table.

Step 2 of redoing obs for an already-shuffled combine (uce.h5 untouched):

  1. python3.11 FM_wrangling/combineEmbedding.py --embedding uce --obs-only --results_dir <tmp_dir>
     -> <tmp_dir>/obs.tsv.gz, rows in the original pre-shuffle order
  2. python3.11 FM_wrangling/apply_shuffle_index.py \
         --obs <tmp_dir>/obs.tsv.gz \
         --shuffle-index <results_dir>/shuffle_index.uce.npy \
         --out <results_dir>/obs.tsv.gz

The pre-shuffle row order is only reproduced if step 1 sees the same datasets,
dataset lists and source files as the original --shuffle run. So before writing
--out, the new new_index column is compared row by row against a reference obs:
--reference-obs if given, else the existing --out file (the obs being replaced).
On any mismatch nothing is replaced.
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np

from obs_shuffle import compare_index_column, obs_tsv_to_parquet, write_shuffled_obs

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--obs", required=True,
                    help="Unshuffled obs.tsv.gz from combineEmbedding.py --obs-only")
parser.add_argument("--shuffle-index", required=True,
                    help="shuffle_index.uce.npy saved by the original combineEmbedding.py --shuffle run")
parser.add_argument("--out", required=True,
                    help="Output path for the shuffled obs.tsv.gz (typically the results_dir obs.tsv.gz)")
parser.add_argument("--reference-obs", default=None,
                    help="Shuffled obs whose new_index order the output must match "
                         "(default: the existing --out file, if any)")
parser.add_argument("--staging-dir", default="/nvme",
                    help="Fast local scratch for the Parquet staging and output (default: /nvme)")
args = parser.parse_args()

reference = args.reference_obs or (args.out if os.path.exists(args.out) else None)
if reference is None:
    print(f"WARNING: no --reference-obs and {args.out} does not exist — only the row "
          f"count can be checked, not that rows line up with uce.h5")
elif not os.path.exists(reference):
    sys.exit(f"ERROR: --reference-obs {reference} does not exist")

shuffle_idx = np.load(args.shuffle_index)
print(f"Shuffle index: {len(shuffle_idx):,} rows from {args.shuffle_index}")

tag = f"{os.getpid()}"
pq_tmp = os.path.join(args.staging_dir, f"apply_shuffle_obs_{tag}.parquet")
out_tmp = os.path.join(args.staging_dir, f"apply_shuffle_obs_{tag}.tsv.gz")

print(f"Staging {args.obs} -> {pq_tmp}")
n_rows = obs_tsv_to_parquet(args.obs, pq_tmp)
print(f"  {n_rows:,} rows")
if n_rows != len(shuffle_idx):
    os.remove(pq_tmp)
    sys.exit(f"ERROR: obs has {n_rows:,} rows but shuffle index has {len(shuffle_idx):,}; "
             f"the --obs-only rerun did not reproduce the original rows")

print(f"Writing shuffled obs -> {out_tmp}")
write_shuffled_obs(pq_tmp, shuffle_idx, out_tmp)
os.remove(pq_tmp)

if reference is not None:
    print(f"Verifying new_index order against {reference} ...")
    diff = compare_index_column(out_tmp, reference)
    if diff is not None:
        sys.exit(f"ERROR: new_index order differs from {reference} ({diff}). "
                 f"{args.out} NOT replaced; new output left at {out_tmp} for inspection.")
    print("  new_index order matches")

print(f"Moving {out_tmp} -> {args.out}")
# copy next to --out, then rename: a cross-filesystem shutil.move would copy straight
# over the existing obs, leaving it truncated if interrupted
out_partial = args.out + ".partial"
shutil.copyfile(out_tmp, out_partial)
os.replace(out_partial, args.out)
os.remove(out_tmp)

notes = os.path.join(os.path.dirname(os.path.abspath(args.out)), "wrangling_notes_apply_shuffle_index.txt")
with open(notes, "a") as f:
    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {args.out} regenerated from {args.obs} "
            f"with shuffle index {args.shuffle_index} ({n_rows:,} rows); new_index order "
            + (f"verified against {reference}\n" if reference else "NOT verified (no reference obs)\n"))
print(f"Done. Notes appended to {notes}")
