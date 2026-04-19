"""
Convert a .npy embedding file to .h5 with a single 'data' dataset.

Usage:
    python3 npy_to_h5.py input.npy [output.h5]

If output path is omitted, saves alongside the input file with .h5 extension.
"""

import sys
import os
import numpy as np
import h5py

def convert(npy_path, h5_path=None):
    if h5_path is None:
        h5_path = os.path.splitext(npy_path)[0] + ".h5"

    print(f"Loading {npy_path} ...")
    arr = np.load(npy_path)
    print(f"  Shape: {arr.shape}  dtype: {arr.dtype}")

    print(f"Writing {h5_path} ...")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=arr, dtype="float32",
                         chunks=True, compression="gzip", compression_opts=4)
    size_gb = os.path.getsize(h5_path) / 1e9
    print(f"  Done. File size: {size_gb:.2f} GB")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    npy_path = sys.argv[1]
    h5_path  = sys.argv[2] if len(sys.argv) > 2 else None
    convert(npy_path, h5_path)
