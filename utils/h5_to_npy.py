"""
Convert an .h5 file (single 'data' dataset) back to a .npy embedding file.

Usage:
    python3 h5_to_npy.py input.h5 [output.npy]

If output path is omitted, saves alongside the input file with .npy extension.
"""

import sys
import os
import numpy as np
import h5py

def convert(h5_path, npy_path=None):
    if npy_path is None:
        npy_path = os.path.splitext(h5_path)[0] + ".npy"

    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        arr = f["data"][:]
    print(f"  Shape: {arr.shape}  dtype: {arr.dtype}")

    print(f"Writing {npy_path} ...")
    np.save(npy_path, arr)
    size_gb = os.path.getsize(npy_path) / 1e9
    print(f"  Done. File size: {size_gb:.2f} GB")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    h5_path  = sys.argv[1]
    npy_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(h5_path, npy_path)
