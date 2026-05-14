"""
shuffle.py

Shuffles an embeddings file (.h5 or .npy) and obs.tsv.gz with a shared random
permutation so that row ordering stays consistent between the two files.

Modes:
  default   — loads the full array into RAM twice (fast, needs 2× data size in RAM)
  --chunked — reads/writes in chunks using sort-then-unsort for sequential HDF5
              access (slow, needs only ~2× chunk_size rows in RAM at a time)
              Only supported for .h5 input/output.

Outputs:
  <uce-out>       — shuffled embeddings (default: <stem>_shuffled.h5/.npy)
  <obs-out>       — shuffled metadata   (default: <stem>_shuffled.tsv.gz)
  <idx-out>       — the permutation array for reproducibility (default: shuffle_index.npy)
"""

import argparse
import numpy as np
import h5py
import pandas as pd
import psutil
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random each run)")
    parser.add_argument("--uce-in",  default="uce.h5",
                        help="Input embeddings file, .h5 or .npy (default: uce.h5)")
    parser.add_argument("--obs-in",  default="obs.tsv.gz",
                        help="Input obs TSV file (default: obs.tsv.gz)")
    parser.add_argument("--uce-out", default=None,
                        help="Output embeddings file (default: <uce-in stem>_shuffled.h5/.npy)")
    parser.add_argument("--obs-out", default=None,
                        help="Output obs TSV file (default: obs_shuffled.tsv.gz)")
    parser.add_argument("--idx-out", default="shuffle_index.npy",
                        help="File to save the permutation index (default: shuffle_index.npy)")
    parser.add_argument("--chunked", action="store_true", default=False,
                        help="Memory-efficient chunked mode (.h5 only). Slower but uses ~2x chunk RAM instead of 2x full array.")
    parser.add_argument("--chunk-size", type=int, default=250_000,
                        help="Rows per chunk in --chunked mode (default: 250000, ~1.3 GB for 1280-dim float32)")
    return parser.parse_args()


def derive_output_names(args):
    read_npy = args.uce_in.endswith(".npy")
    if args.uce_out is None:
        if read_npy:
            args.uce_out = args.uce_in[:-4] + "_shuffled.npy"
        else:
            stem = args.uce_in[:-3] if args.uce_in.endswith(".h5") else args.uce_in
            args.uce_out = stem + "_shuffled.h5"
    if args.obs_out is None:
        stem = args.obs_in[:-7] if args.obs_in.endswith(".tsv.gz") else args.obs_in
        args.obs_out = stem + "_shuffled.tsv.gz"
    return read_npy


def get_array_info(uce_in, read_npy):
    if read_npy:
        tmp = np.load(uce_in, mmap_mode="r")
        shape = tmp.shape
        dtype = tmp.dtype
        del tmp
    else:
        with h5py.File(uce_in, "r") as f:
            shape = f["data"].shape
            dtype = f["data"].dtype
    return shape, dtype


def shuffle_default(args, idx, n_rows, read_npy):
    """Load full array into RAM, shuffle, write."""
    print(f"Reading {args.uce_in} ...")
    if read_npy:
        data = np.load(args.uce_in)
    else:
        with h5py.File(args.uce_in, "r") as f:
            data = f["data"][:]

    data_shuffled = data[idx]
    del data

    print(f"Writing {args.uce_out} ...")
    if args.uce_out.endswith(".npy"):
        np.save(args.uce_out, data_shuffled)
    else:
        with h5py.File(args.uce_out, "w") as f:
            f.create_dataset("data", data=data_shuffled, compression="gzip", compression_opts=4)
    print(f"  shape: {data_shuffled.shape}")


def shuffle_chunked(args, idx, n_rows, n_cols, dtype):
    """Memory-efficient shuffle using sort-then-unsort for sequential HDF5 reads."""
    if not args.uce_in.endswith(".h5"):
        raise ValueError("--chunked mode only supported for .h5 files")

    chunk_size = args.chunk_size
    chunk_bytes = 2 * chunk_size * n_cols * dtype.itemsize
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    print(f"Chunked mode: {n_chunks} chunks of {chunk_size:,} rows  |  ~{chunk_bytes/1024**3:.2f} GB per chunk")

    with h5py.File(args.uce_in, "r") as f_in, h5py.File(args.uce_out, "w") as f_out:
        ds_out = f_out.create_dataset("data", shape=(n_rows, n_cols), dtype=dtype,
                                      compression="gzip", compression_opts=4)
        for i in range(n_chunks):
            out_start = i * chunk_size
            out_end   = min(out_start + chunk_size, n_rows)

            src_rows   = idx[out_start:out_end]          # which input rows feed this output chunk
            sort_order = np.argsort(src_rows)             # sort for sequential HDF5 reads
            sorted_src = src_rows[sort_order]

            data = f_in["data"][sorted_src]               # sequential read
            data = data[np.argsort(sort_order)]           # restore correct output order

            ds_out[out_start:out_end] = data
            print(f"  chunk {i+1}/{n_chunks}: output rows {out_start:,}–{out_end:,}")


def shuffle_obs(args, idx, n_rows):
    print(f"Reading {args.obs_in} ...")
    obs = pd.read_csv(args.obs_in, sep="\t", low_memory=False)
    assert len(obs) == n_rows, (
        f"Row count mismatch: {args.uce_in} has {n_rows} rows but {args.obs_in} has {len(obs)}"
    )
    obs_shuffled = obs.iloc[idx].reset_index(drop=True)
    print(f"Writing {args.obs_out} ...")
    obs_shuffled.to_csv(args.obs_out, sep="\t", index=False, compression="gzip")
    print(f"  shape: {obs_shuffled.shape}")


def sanity_check(args, read_npy):
    print("\nSanity check — first 3 rows of each output:")
    if args.uce_out.endswith(".npy"):
        check = np.load(args.uce_out, mmap_mode="r")
        print(f"  {args.uce_out} data[0:3, 0:5]:\n", check[:3, :5])
    else:
        with h5py.File(args.uce_out, "r") as f:
            print(f"  {args.uce_out} data[0:3, 0:5]:\n", f["data"][:3, :5])
    obs_check = pd.read_csv(args.obs_out, sep="\t", nrows=3, low_memory=False)
    print("  obs_shuffled first column:\n", obs_check.iloc[:, 0].tolist())
    print("\nDone. Row ordering is consistent between both output files.")


def main():
    args = parse_args()
    read_npy = derive_output_names(args)

    shape, dtype = get_array_info(args.uce_in, read_npy)
    n_rows = shape[0]
    n_cols = shape[1] if len(shape) > 1 else 1
    data_gb = n_rows * n_cols * dtype.itemsize / 1024**3
    available_gb = psutil.virtual_memory().available / 1024**3
    print(f"Total rows: {n_rows:,}  |  uncompressed data: {data_gb:.2f} GB  |  Available RAM: {available_gb:.2f} GB")

    if not args.chunked:
        peak_gb = 2 * data_gb
        if peak_gb > available_gb:
            print(f"WARNING: estimated peak RAM ({peak_gb:.2f} GB) exceeds available RAM "
                  f"({available_gb:.2f} GB). Consider --chunked mode.")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_rows)
    np.save(args.idx_out, idx)
    print(f"Permutation saved to {args.idx_out}  (seed={args.seed})")

    if args.chunked:
        shuffle_chunked(args, idx, n_rows, n_cols, dtype)
    else:
        shuffle_default(args, idx, n_rows, read_npy)

    shuffle_obs(args, idx, n_rows)
    sanity_check(args, read_npy)


if __name__ == "__main__":
    main()
