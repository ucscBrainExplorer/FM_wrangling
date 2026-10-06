"""
shuffle.py

Shuffles an embeddings file (.h5 or .npy) and its obs TSV with a shared random
permutation so that row ordering stays consistent between the two files.

Modes:
  default   — loads the full array into RAM twice (fast, needs 2× data size in RAM)
  --chunked — scatter-gather through temp files on --nvme-dir. One sequential pass
              through the source scatters rows into ceil(n_rows / --chunk-size) temp
              chunk files; a second pass concatenates them into the shuffled output.
              RAM: index arrays (~32 bytes/row) + one --read-size block.
              NVMe: temp files totalling the full uncompressed array
              (e.g. 56.66M × 1280 float32 → 227 files, ~290 GB).
              Only supported for .h5 input.
              If --out-dir is on the same disk as --nvme-dir, the output needs a second
              full copy's worth of space; --delete-temp-as-gathered drops each temp file
              once it is copied, so temp + output never exceed one copy.

Input layout matters: an h5 stored in tiny chunks (e.g. the 56M uce.h5, 64×80) turns
the sequential read into millions of small reads, very slow over EFS/NFS. Copy it to
local NVMe first (one large sequential read) and pass that copy as --uce-in.

Obs is reordered with the chunked Arrow path in FM_wrangling/obs_shuffle.py (staged
as all-string Parquet in --nvme-dir), so every value is written back verbatim
("0012" stays "0012", "NA" stays "NA") and RAM stays ~10–20 GB at 50M+ rows.

Outputs (in --out-dir):
  uce.h5             — shuffled embeddings, uncompressed contiguous (as combineEmbedding.py)
  obs.tsv.gz         — shuffled metadata
  shuffle_index.npy  — the permutation: output row i = input row shuffle_index[i].
                       Pass it to apply_shuffle_index.py --shuffle-index to redo obs later.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obs_shuffle import obs_tsv_to_parquet, write_shuffled_obs  # FM_wrangling/obs_shuffle.py


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True,
                        help="Output directory — uce.h5, obs.tsv.gz, shuffle_index.npy written here")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random each run)")
    parser.add_argument("--uce-in",  default="uce.h5",
                        help="Input embeddings file, .h5 or .npy (default: uce.h5)")
    parser.add_argument("--obs-in",  default="obs.tsv.gz",
                        help="Input obs TSV file, plain or .gz (default: obs.tsv.gz)")
    parser.add_argument("--chunked", action="store_true", default=False,
                        help="Memory-efficient chunked mode (.h5 only). One sequential read pass "
                             "scatters rows to NVMe temp files; a second pass gathers into output.")
    parser.add_argument("--chunk-size", type=int, default=250_000,
                        help="Rows per temp chunk file (default: 250000, ~1.3 GB for 1280-dim float32)")
    parser.add_argument("--read-size", type=int, default=50_000,
                        help="Rows per sequential read from source in chunked mode (default: 50000)")
    parser.add_argument("--nvme-dir", default="/nvme",
                        help="Scratch space for temp chunk files and obs Parquet staging (default: /nvme)")
    parser.add_argument("--gather-only", action="store_true", default=False,
                        help="Skip Phase 1 (scatter) and go straight to Phase 2 (gather). "
                             "Use when chunk temp files already exist on --nvme-dir; reuses the "
                             "existing <out-dir>/shuffle_index.npy instead of drawing a new one.")
    parser.add_argument("--delete-temp-as-gathered", action="store_true", default=False,
                        help="Delete each temp chunk file as soon as it is copied into the output, so "
                             "temp + output never exceed one copy of the data. Needed when --out-dir is "
                             "on the same disk as --nvme-dir and space is short. An interrupted gather "
                             "can then NOT be resumed with --gather-only; rerun from scratch.")
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
        f.create_dataset("data", data=data_shuffled)
    print(f"  shape: {data_shuffled.shape}")


def shuffle_chunked(args, idx, n_rows, n_cols, dtype):
    """
    Memory-efficient shuffle via scatter-gather with NVMe temp files.

    Phase 1 — Scatter (one sequential read pass through source):
      For each read block, write every row to its slot in the memmapped temp file
      of the output chunk it belongs to.

    Phase 2 — Gather (sequential read of each temp file):
      Concatenate chunk temp files in order into the final shuffled output H5.
    """
    chunk_size = args.chunk_size
    read_size  = args.read_size
    n_chunks   = (n_rows + chunk_size - 1) // chunk_size
    nvme_dir   = args.nvme_dir
    temp_gb    = n_rows * n_cols * dtype.itemsize / 1024**3

    print(f"Chunked scatter-gather mode:")
    print(f"  {n_chunks} output chunks of {chunk_size:,} rows ({temp_gb:.1f} GB of temp files)")
    print(f"  read block size: {read_size:,} rows")
    print(f"  temp files: {nvme_dir}/chunk_XXXX.npy")

    tmp_paths = [os.path.join(nvme_dir, f"chunk_{i:04d}.npy") for i in range(n_chunks)]
    chunk_sizes_actual = [min(chunk_size, n_rows - i * chunk_size) for i in range(n_chunks)]

    if not args.gather_only:
        free_gb = psutil.disk_usage(nvme_dir).free / 1024**3
        # the output is written sparsely as chunks are gathered; on the same disk it
        # needs its own full copy unless temp files are deleted as they are consumed
        same_disk = os.stat(nvme_dir).st_dev == os.stat(args.out_dir).st_dev
        need_gb = temp_gb * (2 if same_disk and not args.delete_temp_as_gathered else 1)
        if need_gb > free_gb:
            hint = " (add --delete-temp-as-gathered)" if same_disk and not args.delete_temp_as_gathered else ""
            sys.exit(f"ERROR: temp files{' + output' if need_gb > temp_gb else ''} need {need_gb:.1f} GB "
                     f"but {nvme_dir} has {free_gb:.1f} GB free{hint}")

        # inv_idx[src_row] = output position of that source row
        print("Building inverse shuffle index...")
        inv_idx = np.empty(n_rows, dtype=np.int64)
        inv_idx[idx] = np.arange(n_rows, dtype=np.int64)
        chunk_assign = (inv_idx // chunk_size).astype(np.int32)  # which chunk (0..n_chunks-1)
        chunk_pos    = (inv_idx % chunk_size).astype(np.int32)   # position within that chunk
        del inv_idx

        print("  Pre-allocating temp chunk files on NVMe...")
        chunk_mmaps = [np.memmap(p, dtype=dtype, mode="w+", shape=(cs, n_cols))
                       for p, cs in zip(tmp_paths, chunk_sizes_actual)]

        print(f"\nPhase 1 — Scatter: reading {args.uce_in} sequentially...")
        with h5py.File(args.uce_in, "r") as f_in:
            src = f_in["data"]
            n_read_blocks = (n_rows + read_size - 1) // read_size
            for b in range(n_read_blocks):
                src_start = b * read_size
                src_end   = min(src_start + read_size, n_rows)
                block = src[src_start:src_end]  # sequential read

                # group the block's rows by destination chunk: one fancy-indexed
                # write per chunk instead of one Python-level write per row
                c_blk = chunk_assign[src_start:src_end]
                p_blk = chunk_pos[src_start:src_end]
                order = np.argsort(c_blk, kind="stable")
                bounds = np.flatnonzero(np.diff(c_blk[order])) + 1
                for grp in np.split(order, bounds):
                    chunk_mmaps[c_blk[grp[0]]][p_blk[grp]] = block[grp]

                if (b + 1) % 10 == 0 or b == n_read_blocks - 1:
                    print(f"  scatter block {b+1}/{n_read_blocks}: "
                          f"rows {src_start:,}–{src_end:,}", flush=True)

        for m in chunk_mmaps:
            m.flush()
        del chunk_mmaps
    else:
        print("  --gather-only: skipping scatter phase, using existing temp files.")
        missing = [p for p in tmp_paths if not os.path.exists(p)]
        if missing:
            sys.exit(f"ERROR: {len(missing)} temp chunk files missing, e.g. {missing[0]}")

    # Phase 2: gather
    print(f"\nPhase 2 — Gather: writing {args.uce_out}...")
    with h5py.File(args.uce_out, "w") as f_out:
        ds_out = f_out.create_dataset("data", shape=(n_rows, n_cols), dtype=dtype)
        for i, (tmp_path, cs) in enumerate(zip(tmp_paths, chunk_sizes_actual)):
            out_start = i * chunk_size
            out_end   = out_start + cs
            chunk_data = np.memmap(tmp_path, dtype=dtype, mode="r", shape=(cs, n_cols))
            ds_out[out_start:out_end] = chunk_data
            del chunk_data
            if args.delete_temp_as_gathered:
                os.remove(tmp_path)
            print(f"  chunk {i+1}/{n_chunks}: output rows {out_start:,}–{out_end:,}", flush=True)

    # by default, remove temp files only once the output is fully written and closed,
    # so an interrupted gather can be resumed with --gather-only
    if not args.delete_temp_as_gathered:
        for tmp_path in tmp_paths:
            os.remove(tmp_path)
    print("Scatter-gather complete.")


def shuffle_obs(args, idx, n_rows):
    pq_tmp = os.path.join(args.nvme_dir, f"shuffle_obs_{os.getpid()}.parquet")
    print(f"Staging {args.obs_in} -> {pq_tmp} ...")
    n_obs = obs_tsv_to_parquet(args.obs_in, pq_tmp)
    if n_obs != n_rows:
        os.remove(pq_tmp)
        sys.exit(f"ERROR: row count mismatch: {args.uce_in} has {n_rows:,} rows "
                 f"but {args.obs_in} has {n_obs:,}")
    print(f"Writing {args.obs_out} ...")
    write_shuffled_obs(pq_tmp, idx, args.obs_out)
    os.remove(pq_tmp)


def sanity_check(args, idx, read_npy, n_check=5):
    """Spot-check that output row i equals input row idx[i] in both files."""
    print("\nSanity check:")
    rng = np.random.default_rng(0)
    out_rows = np.sort(rng.choice(len(idx), size=min(n_check, len(idx)), replace=False))
    src_rows = idx[out_rows]

    with h5py.File(args.uce_out, "r") as f:
        out_emb = f["data"][out_rows]
    if read_npy:
        src_emb = np.load(args.uce_in, mmap_mode="r")[src_rows]
    else:
        with h5py.File(args.uce_in, "r") as f:
            src_emb = np.stack([f["data"][r] for r in src_rows])
    assert np.array_equal(out_emb, src_emb), "embedding rows do not match source rows"
    print(f"  embeddings: output rows {out_rows.tolist()} == input rows {src_rows.tolist()}")

    # first column of obs (new_index) at the same positions
    src_ids = first_col_at(args.obs_in, src_rows)
    out_ids = first_col_at(args.obs_out, out_rows)
    for o, s in zip(out_rows, src_rows):
        assert out_ids[o] == src_ids[s], f"obs row {o}: {out_ids[o]!r} != input row {s}: {src_ids[s]!r}"
    print(f"  obs first column matches at the same rows")
    print("\nDone. Row ordering is consistent between both output files.")


def first_col_at(path, rows, chunk_rows=1_000_000):
    """Return {row: first-column value} for the given rows of a TSV(.gz), streaming."""
    rows = np.asarray(rows)
    found, start = {}, 0
    for chunk in pd.read_csv(path, sep="\t", usecols=[0], dtype=str,
                             keep_default_na=False, chunksize=chunk_rows):
        for r in rows[(rows >= start) & (rows < start + len(chunk))]:
            found[int(r)] = chunk.iloc[r - start, 0]
        start += len(chunk)
    return found


def main():
    args = parse_args()
    read_npy = derive_output_names(args)

    shape, dtype = get_array_info(args.uce_in, read_npy)
    n_rows = shape[0]
    n_cols = shape[1] if len(shape) > 1 else 1
    data_gb = n_rows * n_cols * dtype.itemsize / 1024**3
    available_gb = psutil.virtual_memory().available / 1024**3
    print(f"Total rows: {n_rows:,}  |  uncompressed data: {data_gb:.2f} GB  |  Available RAM: {available_gb:.2f} GB")

    if args.chunked and read_npy:
        sys.exit("ERROR: --chunked requires .h5 input")
    if not args.chunked:
        peak_gb = 2 * data_gb
        if peak_gb > available_gb:
            print(f"WARNING: estimated peak RAM ({peak_gb:.2f} GB) exceeds available RAM "
                  f"({available_gb:.2f} GB). Consider --chunked mode.")

    if args.gather_only:
        # the temp files were scattered with the saved permutation; a new one would
        # silently misalign obs against the embeddings
        idx = np.load(args.idx_out)
        if len(idx) != n_rows:
            sys.exit(f"ERROR: {args.idx_out} has {len(idx):,} rows, input has {n_rows:,}")
        print(f"--gather-only: reusing permutation from {args.idx_out}")
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(n_rows)
        np.save(args.idx_out, idx)
        print(f"Permutation saved to {args.idx_out}  (seed={args.seed})")

    if args.chunked:
        shuffle_chunked(args, idx, n_rows, n_cols, dtype)
    else:
        shuffle_default(args, idx, n_rows, read_npy)

    shuffle_obs(args, idx, n_rows)
    sanity_check(args, idx, read_npy)


if __name__ == "__main__":
    main()
