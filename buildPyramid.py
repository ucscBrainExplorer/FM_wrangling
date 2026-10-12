"""
buildPyramid.py — build a pyramid tile set from a pyramid config JSON.

Usage:
    python3 FM_wrangling/buildPyramid.py \
        --results_dir HNOCA_Extended_ff9982c2-06cc-4376-930b-186fcbc1fd82/ \
        --pyramid_file pyramid_e30_config.json
"""

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import h5py
import imageio
import numpy as np
import pandas as pd

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True,
                    help="Directory containing the UMAP .npy and metadata TSV referenced by the pyramid config")
parser.add_argument("--pyramid_file", required=True,
                    help="Pyramid config JSON filename inside results_dir (e.g. pyramid_e30_config.json)")
parser.add_argument("--metadata_only", action="store_true",
                    help="Generate metadata.json without building pyramid tiles")
parser.add_argument("--skip_top_levels", type=int, default=0,
                    help="Number of highest-resolution levels to skip (default: 0 = save all)")
args = parser.parse_args()

results_dir  = args.results_dir
pyramid_file = args.pyramid_file

# ── Functions ─────────────────────────────────────────────────────────────────

def scale_to_int_coords(array, target_min, target_max):
    scaled_array = ((array - array.min()) / (array.max() - array.min())) * (target_max - target_min) + target_min
    return np.round(scaled_array).astype(int)


def convert_UMAP(arr1_size, UMAP):
    UMAP1_size = np.max(UMAP[:, 0]) - np.min(UMAP[:, 0])
    UMAP2_size = np.max(UMAP[:, 1]) - np.min(UMAP[:, 1])
    arr2_size = int(UMAP2_size / UMAP1_size * arr1_size)

    UMAP_df = pd.DataFrame(np.zeros((UMAP.shape[0], 4)),
                           columns=["UMAP1_scaleInt", "UMAP2_scaleInt", "tile_x", "tile_y"])
    UMAP_df["UMAP1_scaleInt"] = scale_to_int_coords(UMAP[:, 0], 0, arr1_size)
    UMAP_df["UMAP2_scaleInt"] = scale_to_int_coords(UMAP[:, 1], 0, arr2_size)
    UMAP_df["tile_x"] = UMAP_df["UMAP1_scaleInt"] // tilesize
    UMAP_df["tile_y"] = UMAP_df["UMAP2_scaleInt"] // tilesize
    return UMAP_df.to_numpy()


def UMAP_loss_rate(arr1_size, UMAP):
    UMAP1_size = np.max(UMAP[:, 0]) - np.min(UMAP[:, 0])
    UMAP2_size = np.max(UMAP[:, 1]) - np.min(UMAP[:, 1])
    arr2_size = int(UMAP2_size / UMAP1_size * arr1_size)

    UMAP_df = pd.DataFrame(np.zeros((UMAP.shape[0], 4)),
                           columns=["UMAP1_scaleInt", "UMAP2_scaleInt", "tile_x", "tile_y"])
    UMAP_df["UMAP1_scaleInt"] = scale_to_int_coords(UMAP[:, 0], 0, arr1_size)
    UMAP_df["UMAP2_scaleInt"] = scale_to_int_coords(UMAP[:, 1], 0, arr2_size)
    UMAP_df["tile_x"] = UMAP_df["UMAP1_scaleInt"] // tilesize
    UMAP_df["tile_y"] = UMAP_df["UMAP2_scaleInt"] // tilesize

    pixel_counts = UMAP_df.groupby(["UMAP1_scaleInt", "UMAP2_scaleInt"]).size()
    tile_counts  = UMAP_df.groupby(["tile_x", "tile_y"]).size()
    total_tile   = np.max(UMAP_df["tile_x"]) * np.max(UMAP_df["tile_y"])
    avg_points_per_non_empty_tile = tile_counts.mean()

    loss_rate = 1 - len(pixel_counts) / UMAP_df.shape[0]
    print(arr1_size, arr2_size, len(pixel_counts), len(tile_counts), total_tile,
          avg_points_per_non_empty_tile, loss_rate)
    return loss_rate


def findCodes(data, feature, order="alphabetical"):
    print(feature)
    if order == "frequency":
        # The viewer cycles a small palette over the codes, so categories whose
        # codes differ by a multiple of the palette size share a color. Coding by
        # descending cell count gives the largest categories the first codes,
        # which are distinct colors whatever the palette size is.
        counts = data[feature].value_counts()
        classes = sorted(counts.index, key=lambda c: (-counts[c], str(c)))
        categories_int = pd.Categorical(data[feature], categories=classes).codes.astype(int) + 1
    else:
        from sklearn.preprocessing import LabelEncoder
        categories = data[feature].to_numpy()
        le = LabelEncoder()
        mask = ~pd.isna(categories)
        categories_int = np.zeros_like(categories, dtype=int)
        categories_int[mask] = le.fit_transform(categories[mask]) + 1
        classes = le.classes_
    cat_list = ["background"] + [str(int(c)) if isinstance(c, float) and c.is_integer() else str(c) for c in classes]
    print(f"  {len(classes)} categories ({order} order)")
    return categories_int, cat_list


def binContinuous(data, feature, n_bins=10):
    values = data[feature].to_numpy(dtype=float)
    categories_int = np.zeros(len(values), dtype=int)
    mask = ~np.isnan(values)
    valid_values = values[mask]

    vmin, vmax = valid_values.min(), valid_values.max()
    step_raw = (vmax - vmin) / n_bins
    magnitude = 10 ** math.floor(math.log10(step_raw))
    nice_step = next(m * magnitude for m in [1, 2, 2.5, 5, 10] if m * magnitude >= step_raw)
    nice_min = math.floor(vmin / nice_step) * nice_step
    nice_max = math.ceil(vmax / nice_step) * nice_step
    actual_n_bins = round((nice_max - nice_min) / nice_step)
    bins = np.array([nice_min + i * nice_step for i in range(actual_n_bins + 1)])

    bin_indices = np.digitize(valid_values, bins)
    bin_indices = np.clip(bin_indices, 1, actual_n_bins)

    bin_counts = np.bincount(bin_indices, minlength=actual_n_bins + 1)[1:]
    non_empty = np.where(bin_counts > 0)[0]

    if nice_step >= 1:
        fmt = ".0f"
    else:
        n_dec = max(1, -int(math.floor(math.log10(nice_step))))
        fmt = f".{n_dec}f"

    all_labels  = [f"{bins[i]:{fmt}}–{bins[i+1]:{fmt}}" for i in range(actual_n_bins)]
    kept_labels = [all_labels[i] for i in non_empty]
    old_to_new  = {old + 1: new + 1 for new, old in enumerate(non_empty)}
    categories_int[mask] = np.vectorize(old_to_new.get)(bin_indices)

    cat_list = ["background"] + kept_labels
    print(f"{feature}: {len(kept_labels)} bins (of {actual_n_bins}), {mask.sum()} valid / {len(values)} total")
    return categories_int, cat_list


def find_depth_bit(cat_list):
    return 8 if len(cat_list) <= 256 else 16


def determine_maxLevel(h, w):
    max_s = max(h, w)
    return int(math.ceil(math.log2(max_s)) - math.log2(tilesize))


def get_umap_df_dim(UMAP_df):
    x, y = UMAP_df[:, 0], UMAP_df[:, 1]
    height = ((y.max() + tilesize) // tilesize) * tilesize
    width  = ((x.max() + tilesize) // tilesize) * tilesize
    return height, width


def build_array_2d_cat(UMAP_df, tilesize):
    x, y = UMAP_df[:, 0], UMAP_df[:, 1]
    height = ((y.max() + tilesize) // tilesize) * tilesize
    width  = ((x.max() + tilesize) // tilesize) * tilesize
    print(f"Final shape: h {height} x w {width}")

    max_level = determine_maxLevel(height, width)
    print(f"Levels: {max_level + 1}  Max Level: {max_level}")

    UMAP_cat = np.hstack([UMAP_df, categories_int.reshape(-1, 1)])
    keys = UMAP_cat[:, :2]

    unique_keys_first, pixel_totalcount = np.unique(keys, axis=0, return_counts=True)

    cats = UMAP_cat[:, 4]
    sorted_idx = np.lexsort((keys[:, 0], keys[:, 1]))
    keys_sorted = keys[sorted_idx]
    cats_sorted = cats[sorted_idx]
    unique_keys_second, start_idx, counts = np.unique(keys_sorted, axis=0,
                                                       return_index=True, return_counts=True)
    assert np.array_equal(unique_keys_first, unique_keys_second)

    np.random.seed(42)
    random_offsets = np.random.randint(0, counts)
    pixel_cats = cats_sorted[start_idx + random_offsets]

    array_2d_cat = np.concatenate(
        [unique_keys_first, unique_keys_first // tilesize,
         pixel_totalcount[:, None], pixel_cats[:, None]], axis=1
    )
    return array_2d_cat, max_level


def downsample_categorical_by_frequency(arr, pixel_area_equivalent, categories_size,
                                        density_threshold, block_size=2, background_value=0):
    new_xy = arr[:, :2] // block_size
    counts_arr = arr[:, 4]
    cats_arr   = arr[:, 5]

    unique_keys_first, inverse_idx = np.unique(new_xy, axis=0, return_inverse=True)
    pixel_totalcount = np.bincount(inverse_idx, weights=counts_arr).astype(int)

    sorted_idx = np.lexsort((new_xy[:, 0], new_xy[:, 1]))
    keys_sorted = new_xy[sorted_idx]
    cats_sorted = cats_arr[sorted_idx]
    unique_keys_second, start_idx, counts = np.unique(keys_sorted, axis=0,
                                                       return_index=True, return_counts=True)
    assert np.array_equal(unique_keys_first, unique_keys_second)

    random_offsets = np.random.randint(0, counts)
    pixel_cats = cats_sorted[start_idx + random_offsets]

    mask_count = pixel_totalcount / pixel_area_equivalent < density_threshold
    rand_mask  = np.random.rand(len(pixel_totalcount)) < 0.5
    pixel_cats[mask_count & rand_mask] = 0

    downsampled = np.concatenate(
        [unique_keys_first, unique_keys_first // tilesize,
         pixel_totalcount[:, None], pixel_cats[:, None]], axis=1
    )
    return downsampled


def saveAsTiles(array_2d_cat, level, tile_size, bitdepth, start_chn, build_dir, target_dir):
    print(f"Building tiles for level {level}")
    tiles_keys, inverse_idx = np.unique(array_2d_cat[:, 2:4], axis=0, return_inverse=True)

    def process_tile(k_tile):
        k, (tx, ty) = k_tile
        rows   = np.where(inverse_idx == k)[0]
        local_x = array_2d_cat[rows, 0] - tx * tile_size
        local_y = array_2d_cat[rows, 1] - ty * tile_size
        cats    = array_2d_cat[rows, 5]

        dtype = np.uint16 if bitdepth == 16 else np.uint8
        tile  = np.zeros((tile_size, tile_size), dtype=dtype)
        tile[local_y, local_x] = cats

        raw_png   = f"{build_dir}/tile_{level}_{tx*tile_size}_{ty*tile_size}.png"
        final_png = f"{target_dir}/p{start_chn}-{level}-{ty}-{tx}.png"
        imageio.imwrite(raw_png, tile)
        dim = f"{tile_size}x{tile_size}"
        os.system(f"convert {raw_png} -crop {dim} -define png:bit-depth={bitdepth} "
                  f"-depth {bitdepth} -background none -extent {dim} -alpha off {final_png}")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(process_tile, enumerate(tiles_keys))


def write_json(reference_name, viz_label, start_chn, cat_list, feature_type,
               tilesize, max_level, offset_x, offset_y, image_scalef):
    viz_type = "ordinal" if feature_type == "continuous" else "category"
    path = os.path.join(target_dir, "metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            J = json.load(f)
        assert J["tileSize"] == tilesize
        assert J["levels"] == max_level + 1
        assert J["offset"] == [float(offset_x), float(offset_y)]
        assert J["image_scalef"] == float(image_scalef)
        for i in range(len(J["phenotypes"]), start_chn + 1):
            J["phenotypes"].append({})
        J["phenotypes"][start_chn] = {"name": viz_label, "type": viz_type,
                                       "int_to_category": cat_list}
    else:
        height, width = get_umap_df_dim(UMAP_df)
        J = {
            "reference_name": reference_name,
            "count": UMAP.shape[0],
            "phenotypes": [],
            "size": [int(width / math.pow(2, max_level)), int(height / math.pow(2, max_level))],
            "tileSize": tilesize,
            "levels": max_level + 1,
            "fileformat": "png",
            "offset": [float(offset_x), float(offset_y)],
            "image_scalef": float(image_scalef),
        }
        while len(J["phenotypes"]) <= start_chn:
            J["phenotypes"].append({})
        J["phenotypes"][start_chn] = {"name": viz_label, "type": viz_type,
                                       "int_to_category": cat_list}
    print(J)
    with open(path, "w") as f:
        json.dump(J, f, indent=4)


# ── Load config ───────────────────────────────────────────────────────────────

config_path = os.path.join(results_dir, pyramid_file)
with open(config_path) as f:
    config = json.load(f)

reference_name = config["reference_name"]
target_dir     = config["target_dir"]
if not os.path.isabs(target_dir):
    target_dir = os.path.join(results_dir, target_dir)
umap_file      = config["umap_file"]
metadata_file  = config["metadata_file"]

features       = config["features"]
feature_labels = config["feature_labels"]
start_chns     = config.get("start_chns", list(range(len(features))))
feature_types  = config.get("feature_types", ["category"] * len(features))
n_bins_config  = config.get("n_bins", 10)
n_bins_list    = [n_bins_config] * len(features) if isinstance(n_bins_config, int) else n_bins_config
# "alphabetical" (default) or "frequency" — how category codes, and so colors, are assigned
order_config   = config.get("category_order", "alphabetical")
category_orders = [order_config] * len(features) if isinstance(order_config, str) else order_config

assert isinstance(features, list) and isinstance(feature_labels, list)
assert isinstance(start_chns, list) and isinstance(feature_types, list)
assert len(features) == len(feature_labels) == len(feature_types) == len(n_bins_list) == len(category_orders)
assert all(o in ("alphabetical", "frequency") for o in category_orders)
assert all(i < len(features) for i in start_chns)

features       = [features[i]       for i in start_chns]
feature_labels = [feature_labels[i] for i in start_chns]
feature_types  = [feature_types[i]  for i in start_chns]
n_bins_list    = [n_bins_list[i]    for i in start_chns]
category_orders = [category_orders[i] for i in start_chns]

tilesize = 1024

# Use a temp dir per run so parallel builds don't collide
build_dir = tempfile.mkdtemp(prefix="pyramid_build_")
print(f"Build dir: {build_dir}")

os.makedirs(target_dir, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────

print(f"Reading {metadata_file} ...")
# Only `features` columns are ever read from `data` (see findCodes/binContinuous below).
# low_memory=False on the full obs file buffers every column (not just these) while
# sniffing dtypes, which can blow up memory on a large obs file. Instead sniff dtypes
# from a cheap sample restricted to the needed columns, then do a single-pass typed
# read. Widen sampled ints to float64 in case a NaN shows up outside the sample.
metadata_path = os.path.join(results_dir, metadata_file)
usecols = sorted(set(features))
sample = pd.read_csv(metadata_path, sep="\t", usecols=usecols, nrows=200_000)
dtype_map = {
    c: ("float64" if pd.api.types.is_integer_dtype(sample[c]) else sample[c].dtype)
    for c in sample.columns
}
del sample
data = pd.read_csv(metadata_path, sep="\t", usecols=usecols, dtype=dtype_map)

print(f"Loading UMAP {umap_file} ...")
UMAP = np.load(os.path.join(results_dir, umap_file))
print(f"UMAP shape: {UMAP.shape}")

# ── Determine tile dimensions ─────────────────────────────────────────────────

print("arr1_size arr2_size pixel_counts tile_counts total_tile avg_points_per_non_empty_tile loss_rate")
for arr1_size in [2**i for i in range(13, 20)]:
    loss_rate = UMAP_loss_rate(arr1_size, UMAP)
    if loss_rate < 0.03:
        break
print(f"Selected arr1_size: {arr1_size}")

UMAP_df = convert_UMAP(arr1_size, UMAP)

# Compute scale / offset
array      = UMAP[:, 0]
target_max = arr1_size
target_min = 0
image_scalef = 1 / (array.max() - array.min()) * (target_max - target_min)
offset_x     = -array.min() / (array.max() - array.min()) * (target_max - target_min) + target_min
array      = UMAP[:, 1]
target_max = np.max(UMAP_df[:, 1])
offset_y   = -array.min() / (array.max() - array.min()) * (target_max - target_min) + target_min
print(f"image_scalef: {image_scalef}  offset_x: {offset_x}  offset_y: {offset_y}")

# ── Build tiles ───────────────────────────────────────────────────────────────

height, width = get_umap_df_dim(UMAP_df)
max_level = determine_maxLevel(height, width)

for feature, start_chn, viz_label, feature_type, n_bins, category_order in zip(
        features, start_chns, feature_labels, feature_types, n_bins_list, category_orders):

    print(f"\n=== {feature} (chn {start_chn}) ===")
    if feature_type == "continuous":
        categories_int, cat_list = binContinuous(data, feature, n_bins)
    else:
        categories_int, cat_list = findCodes(data, feature, category_order)

    depth_bit = find_depth_bit(cat_list)

    if not args.metadata_only:
        array_2d_cat, max_level = build_array_2d_cat(UMAP_df, tilesize)

        # density threshold lookup
        init_density_threshold = {
            4: 1.0 / 4.0,
            5: 1.0 / 4.0,
            6: 1.0 / 4.0,
            7: 1.0 / (4.0 * 2.0),
            8: 1.0 / (4.0 * 4.0),
            9: 1.0 / (4.0 * 4.0 * 4.0),
            10: 1.0 / (4.0 * 4.0 * 4.0 * 2.0),
        }.get(max_level)

        if init_density_threshold is None:
            print(f"WARNING: no density threshold defined for max_level={max_level}")

        level           = max_level
        current         = array_2d_cat
        current_base_area = 1
        block_size      = 2
        step            = 2.0
        threshold_level = max_level - args.skip_top_levels + 1
        next_base_area  = current_base_area * block_size * block_size
        next_threshold  = init_density_threshold / step
        categories_size = len(cat_list) - 1

        print("Begin downsampling ...")
        while level > 0:
            result = downsample_categorical_by_frequency(
                current, next_base_area, categories_size, next_threshold, block_size=block_size)
            level -= 1
            print(f"Downsampled to level: {level}  count: {result.shape}")

            if level < threshold_level:
                saveAsTiles(result, level, tilesize, depth_bit, start_chn, build_dir, target_dir)
                print(f"Saved tiles, level {level}")

            current        = result
            next_base_area = next_base_area * block_size * block_size
            next_threshold = next_threshold / step

        if args.skip_top_levels == 0:
            saveAsTiles(array_2d_cat, max_level, tilesize, depth_bit, start_chn, build_dir, target_dir)

    write_json(reference_name, viz_label, start_chn, cat_list, feature_type,
               tilesize, max_level, offset_x, offset_y, image_scalef)

# ── Cleanup ───────────────────────────────────────────────────────────────────
shutil.rmtree(build_dir, ignore_errors=True)
print(f"\nDone. Tiles written to {target_dir}")
