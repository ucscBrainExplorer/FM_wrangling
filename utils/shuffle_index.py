"""
shuffle_index.py

Generates a random permutation of row indices, then applies it to both
the embeddings file (uce.h5 or uce.npy) and obs.tsv.gz so that row
ordering stays consistent between the two files.

Supported embedding formats:
  .h5   — HDF5 file with a "data" dataset
  .npy  — NumPy binary array

Outputs:
  uce_shuffled.h5/.npy  — shuffled embeddings (same format as input)
  obs_shuffled.tsv.gz   — shuffled metadata
  shuffle_index.npy     — the permutation array (so the shuffle is reproducible)

Memory requirement:
  The entire dataset is loaded into RAM twice simultaneously — once for
  the original array and once for the shuffled copy — so peak usage is:

      peak RAM ≈ 2 × (n_rows × n_cols × bytes_per_element)

  For float32 data this is:  2 × n_rows × n_cols × 4 bytes

  Examples (float32):
    100 k rows × 1280 cols  →  ~1.0 GB  →  needs ~2 GB RAM
    277 k rows × 1280 cols  →  ~2.8 GB  →  needs ~6 GB RAM  (with OS + pandas overhead)
      1 M rows × 1280 cols  →  ~10 GB   →  needs ~20 GB RAM

  Rule of thumb: available RAM should be > 2× the uncompressed data size.
  .h5 files are gzip-compressed on disk and may be much smaller than the in-memory
  size. .npy files are uncompressed, so on-disk size ≈ in-memory size.
  Always check the actual array shape and dtype, not just the file size.
"""

import argparse
import numpy as np
import h5py
import pandas as pd
import psutil

def main():
    parser = argparse.ArgumentParser(description="Shuffle an embeddings file (.h5 or .npy) and obs.tsv.gz with a shared random index.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random each run)")
    parser.add_argument("--uce-in",  default="uce.h5",
                        help="Input embeddings file, .h5 or .npy (default: uce.h5)")
    parser.add_argument("--obs-in",  default="obs.tsv.gz",
                        help="Input obs TSV file (default: obs.tsv.gz)")
    parser.add_argument("--uce-out", default=None,
                        help="Output embeddings file (default: <uce-in stem>_shuffled.h5/.npy)")
    parser.add_argument("--obs-out", default=None,
                        help="Output obs TSV file (default: <obs-in stem>_shuffled.tsv.gz)")
    parser.add_argument("--idx-out", default="shuffle_index.npy",
                        help="File to save the permutation index (default: shuffle_index.npy)")
    args = parser.parse_args()

    read_npy = args.uce_in.endswith(".npy")

    # Derive output names from input names if not specified
    if args.uce_out is None:
        if read_npy:
            stem = args.uce_in[:-4]
            args.uce_out = stem + "_shuffled.npy"
        else:
            stem = args.uce_in[:-3] if args.uce_in.endswith(".h5") else args.uce_in
            args.uce_out = stem + "_shuffled.h5"

    # Output format follows --uce-out extension; default to input format
    if args.uce_out.endswith(".npy"):
        write_npy = True
    elif args.uce_out.endswith(".h5"):
        write_npy = False
    else:
        raise ValueError(f"--uce-out must end in .npy or .h5, got: {args.uce_out}")

    if args.obs_out is None:
        stem = args.obs_in[:-7] if args.obs_in.endswith(".tsv.gz") else args.obs_in
        args.obs_out = stem + "_shuffled.tsv.gz"

    # ── 1. Determine number of rows from the embeddings file ────────────────
    if read_npy:
        tmp = np.load(args.uce_in, mmap_mode="r")
        n_rows, *rest_dims = tmp.shape
        itemsize = tmp.dtype.itemsize
        del tmp
    else:
        with h5py.File(args.uce_in, "r") as f:
            ds = f["data"]
            n_rows, *rest_dims = ds.shape
            itemsize = ds.dtype.itemsize
    n_elements = n_rows * int(np.prod(rest_dims)) if rest_dims else n_rows
    data_bytes = n_elements * itemsize
    peak_gb = 2 * data_bytes / 1024**3
    available_gb = psutil.virtual_memory().available / 1024**3
    print(f"Total rows: {n_rows:,}  |  uncompressed data: {data_bytes/1024**3:.2f} GB")
    print(f"Peak RAM needed (2× data): {peak_gb:.2f} GB  |  Available RAM: {available_gb:.2f} GB")
    if peak_gb > available_gb:
        print(f"WARNING: estimated peak RAM ({peak_gb:.2f} GB) exceeds available RAM "
              f"({available_gb:.2f} GB). This may cause an out-of-memory crash.")

    # ── 2. Generate (or restore) the permutation ─────────────────────────────
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_rows)
    np.save(args.idx_out, idx)
    print(f"Permutation saved to {args.idx_out}  (seed={args.seed})")

    # ── 3. Shuffle embeddings file ───────────────────────────────────────────
    print(f"Reading {args.uce_in} ...")
    if read_npy:
        data = np.load(args.uce_in)
    else:
        with h5py.File(args.uce_in, "r") as f_in:
            data = f_in["data"][:]      # load into memory

    data_shuffled = data[idx]

    print(f"Writing {args.uce_out} ...")
    if write_npy:
        np.save(args.uce_out, data_shuffled)
    else:
        with h5py.File(args.uce_out, "w") as f_out:
            f_out.create_dataset("data", data=data_shuffled,
                                 compression="gzip", compression_opts=4)
    print(f"  shape: {data_shuffled.shape}")

    # ── 4. Shuffle obs.tsv.gz ────────────────────────────────────────────────
    print(f"Reading {args.obs_in} ...")
    obs = pd.read_csv(args.obs_in, sep="\t", low_memory=False)
    assert len(obs) == n_rows, (
        f"Row count mismatch: {args.uce_in} has {n_rows} rows but {args.obs_in} has {len(obs)}"
    )

    obs_shuffled = obs.iloc[idx].reset_index(drop=True)

    print(f"Writing {args.obs_out} ...")
    obs_shuffled.to_csv(args.obs_out, sep="\t", index=False, compression="gzip")
    print(f"  shape: {obs_shuffled.shape}")

    # ── 5. Sanity check ──────────────────────────────────────────────────────
    print("\nSanity check — first 3 rows of each output:")
    if write_npy:
        check_data = np.load(args.uce_out, mmap_mode="r")
        print(f"  {args.uce_out} data[0:3, 0:5]:\n", check_data[:3, :5])
    else:
        with h5py.File(args.uce_out, "r") as f:
            print(f"  {args.uce_out} data[0:3, 0:5]:\n", f["data"][:3, :5])
    obs_check = pd.read_csv(args.obs_out, sep="\t", nrows=3, low_memory=False)
    print("  obs_shuffled first column (new_index):\n", obs_check.iloc[:, 0].tolist())

    print("\nDone. Row ordering is consistent between both output files.")


if __name__ == "__main__":
    main()
