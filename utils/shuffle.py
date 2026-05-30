"""
shuffle.py

Shuffles an embeddings file (.h5 or .npy) and obs.tsv.gz with a shared random
permutation so that row ordering stays consistent between the two files.

Modes:
  default   — loads the full array into RAM twice (fast, needs 2× data size in RAM)
  --chunked — sequential scatter-gather using temp files on --nvme_dir scratch space.
              One sequential pass through the source writes rows to 26 temp chunk files
              on NVMe; a second pass concatenates them into the shuffled output.
              RAM usage: ~one read buffer (~2.4 GB). NVMe usage: ~33 GB temp files.
              Works efficiently on EFS (sequential reads only).
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
    parser.add_argument("--out-dir", required=True,
                        help="Output directory — uce.h5/.npy, obs.tsv.gz, shuffle_index.npy written here")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random each run)")
    parser.add_argument("--uce-in",  default="uce.h5",
                        help="Input embeddings file, .h5 or .npy (default: uce.h5)")
    parser.add_argument("--obs-in",  default="obs.tsv.gz",
                        help="Input obs TSV file (default: obs.tsv.gz)")
    parser.add_argument("--chunked", action="store_true", default=False,
                        help="Memory-efficient chunked mode (.h5 only). One sequential read pass "
                             "scatters rows to NVMe temp files; a second pass gathers into output.")
    parser.add_argument("--chunk-size", type=int, default=250_000,
                        help="Rows per output chunk (default: 250000, ~1.3 GB for 1280-dim float32)")
    parser.add_argument("--read-size", type=int, default=50_000,
                        help="Rows per sequential read from source in chunked mode (default: 50000)")
    parser.add_argument("--nvme-dir", default="/nvme",
                        help="Scratch space for temp chunk files in --chunked mode (default: /nvme)")
    parser.add_argument("--gather-only", action="store_true", default=False,
                        help="Skip Phase 1 (scatter) and go straight to Phase 2 (gather). "
                             "Use when chunk temp files already exist on --nvme-dir.")
    return parser.parse_args()


def derive_output_names(args):
    os.makedirs(args.out_dir, exist_ok=True)
    read_npy = args.uce_in.endswith(".npy")
    args.uce_out = os.path.join(args.out_dir, "uce.h5")
    args.obs_out = os.path.join(args.out_dir, "obs.tsv.gz")
    args.idx_out = os.path.join(args.out_dir, "shuffle_index.npy")
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
    with h5py.File(args.uce_out, "w") as f:
        f.create_dataset("data", data=data_shuffled, compression="gzip", compression_opts=4)
    print(f"  shape: {data_shuffled.shape}")


def shuffle_chunked(args, idx, n_rows, n_cols, dtype):
    """
    Memory-efficient shuffle via scatter-gather with NVMe temp files.

    Phase 1 — Scatter (one sequential read pass through source):
      For each read block, determine which output chunk each row belongs to
      and append it to that chunk's temp file on NVMe.

    Phase 2 — Gather (sequential read of each temp file):
      Concatenate chunk temp files in order into the final shuffled output H5.
    """
    chunk_size = args.chunk_size
    read_size  = args.read_size
    n_chunks   = (n_rows + chunk_size - 1) // chunk_size
    nvme_dir   = args.nvme_dir

    print(f"Chunked scatter-gather mode:")
    print(f"  {n_chunks} output chunks of {chunk_size:,} rows")
    print(f"  read block size: {read_size:,} rows")
    print(f"  temp files: {nvme_dir}/chunk_XXXX.npy")

    # build inverse index: for each source row, which output chunk does it go to?
    # inv_idx[src_row] = output_position
    print("Building inverse shuffle index...")
    inv_idx = np.empty(n_rows, dtype=np.int64)
    inv_idx[idx] = np.arange(n_rows, dtype=np.int64)
    # chunk assignment for each source row
    chunk_assign = inv_idx // chunk_size  # which chunk (0..n_chunks-1)
    chunk_pos    = inv_idx % chunk_size   # position within that chunk

    # Phase 1: scatter
    print(f"\nPhase 1 — Scatter: reading {args.uce_in} sequentially...")
    # pre-allocate chunk buffers (one at a time to save RAM)
    # use temp .npy files per chunk
    tmp_paths = [os.path.join(nvme_dir, f"chunk_{i:04d}.npy") for i in range(n_chunks)]
    chunk_sizes_actual = [min(chunk_size, n_rows - i * chunk_size) for i in range(n_chunks)]

    # pre-create temp files as memmaps (skip if gather-only — files already exist)
    chunk_mmaps = []
    if not args.gather_only:
        print("  Pre-allocating temp chunk files on NVMe...")
        for i, (tmp_path, cs) in enumerate(zip(tmp_paths, chunk_sizes_actual)):
            m = np.memmap(tmp_path, dtype=dtype, mode="w+", shape=(cs, n_cols))
            chunk_mmaps.append(m)

    if not args.gather_only:
        with h5py.File(args.uce_in, "r") as f_in:
            src = f_in["data"]
            n_read_blocks = (n_rows + read_size - 1) // read_size
            for b in range(n_read_blocks):
                src_start = b * read_size
                src_end   = min(src_start + read_size, n_rows)
                block = src[src_start:src_end]  # sequential read

                # scatter rows to their output chunks
                for local_i, src_row in enumerate(range(src_start, src_end)):
                    c  = chunk_assign[src_row]
                    p  = chunk_pos[src_row]
                    chunk_mmaps[c][p] = block[local_i]

                if (b + 1) % 10 == 0 or b == n_read_blocks - 1:
                    print(f"  scatter block {b+1}/{n_read_blocks}: "
                          f"rows {src_start:,}–{src_end:,}")

        # flush all mmaps
        for m in chunk_mmaps:
            del m
    else:
        print("  --gather-only: skipping scatter phase, using existing temp files.")

    # Phase 2: gather
    print(f"\nPhase 2 — Gather: writing {args.uce_out}...")
    with h5py.File(args.uce_out, "w") as f_out:
        ds_out = f_out.create_dataset("data", shape=(n_rows, n_cols), dtype=dtype,
                                      compression="gzip", compression_opts=4)
        for i, (tmp_path, cs) in enumerate(zip(tmp_paths, chunk_sizes_actual)):
            out_start = i * chunk_size
            out_end   = out_start + cs
            chunk_data = np.memmap(tmp_path, dtype=dtype, mode="r", shape=(cs, n_cols))
            ds_out[out_start:out_end] = chunk_data
            del chunk_data
            os.remove(tmp_path)
            print(f"  chunk {i+1}/{n_chunks}: output rows {out_start:,}–{out_end:,}")

    print("Scatter-gather complete.")


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
