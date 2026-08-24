import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--embedding",
    required=True,
    choices=["uce", "scimilarity"],
    help="Embedding type to use (uce or scimilarity)"
)

parser.add_argument(
    "--results_dir",
    required=True,
    help="Directory path where the output results will be saved"
)

parser.add_argument(
    "--select-datasets-file",
    default=None,
    help="Path to a file with dataset IDs to process (one per line)"
)
parser.add_argument(
    "--obs-only",
    action="store_true",
    default=False,
    help="Only process obs metadata, skip embedding files"
)
parser.add_argument(
    "--uce-only",
    action="store_true",
    default=False,
    help="Only process embedding files, skip obs metadata"
)
parser.add_argument(
    "--shuffle",
    action="store_true",
    default=False,
    help="Shuffle cells in the combined output. Must not be used with --obs-only or --uce-only."
)
parser.add_argument(
    "--nvme-dir",
    default="/nvme",
    help="Path to fast local NVMe scratch space for staging output files before moving to EFS (default: /nvme)"
)

args = parser.parse_args()

import cellxgene_census  # python3.11
import os, sys, re

if args.shuffle and (args.obs_only or args.uce_only):
    print("ERROR: --shuffle cannot be used with --obs-only or --uce-only — obs and uce must be written together to stay in sync.")
    sys.exit(1)

shuffle = args.shuffle
embedding = args.embedding
results_dir = args.results_dir
nvme_dir = args.nvme_dir
import shutil
import ctypes, ctypes.util
import numpy as np
import pandas as pd
import h5py

# After reading a mmap chunk, tell the kernel it can reclaim those pages.
# This keeps RSS flat when sequentially scanning a large read-only mmap.
_libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
_MADV_DONTNEED = 4
_PAGE_SIZE = 4096

def _madvise_dontneed(arr, start_row, end_row):
    stride    = arr.strides[0]
    start_ptr = arr.ctypes.data + start_row * stride
    end_ptr   = arr.ctypes.data + end_row   * stride
    pg_start  = (start_ptr // _PAGE_SIZE) * _PAGE_SIZE
    pg_end    = ((end_ptr + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE
    _libc.madvise(ctypes.c_void_p(pg_start),
                  ctypes.c_size_t(pg_end - pg_start),
                  ctypes.c_int(_MADV_DONTNEED))


def _zip_strict_chunks(uce_row_count, chunk_size, obs_reader, mapped_reader, dataset_id):
    """Like zip(range(...), obs_reader, mapped_reader), but raises instead of silently
    truncating at the shortest iterable when row counts disagree. Plain zip() would
    otherwise drop trailing rows (or misalign obs/mapped pairing) with no error if
    obs.tsv or the remap file don't have the same row count as the .npy embedding."""
    _MISSING = object()
    uce_iter = iter(range(0, uce_row_count, chunk_size))
    while True:
        uce_start = next(uce_iter, _MISSING)
        obs_chunk = next(obs_reader, _MISSING)
        mapped_chunk = (next(mapped_reader, _MISSING)
                        if mapped_reader is not None else None)
        vals = [uce_start, obs_chunk] + ([mapped_chunk] if mapped_reader is not None else [])
        if all(v is _MISSING for v in vals):
            return
        if any(v is _MISSING for v in vals):
            names = ["embedding (.npy)", "obs.tsv"] + (
                ["remap/obs_mapped.tsv.gz"] if mapped_reader is not None else [])
            ran_out = [n for n, v in zip(names, vals) if v is _MISSING]
            raise ValueError(
                f"{dataset_id}: row count mismatch between embedding/obs/remap files — "
                f"{', '.join(ran_out)} ran out of rows first while the others still had "
                f"data. These files must have the same row count and order."
            )
        yield uce_start, obs_chunk, mapped_chunk

obs_only = args.obs_only
uce_only = args.uce_only

if embedding =="uce":
    embedding_outfile = "uce.h5"
    data_dirs =["UCE_npy_obs", "duplicate_UCE"]
    dup_dir = "duplicate_UCE"
    n_cols = 1280
elif embedding == "scimilarity":
    embedding_outfile = "scimilarity.h5"
    data_dirs =["SCimilarity_npy_obs", "duplicate_SCimilarity"]
    dup_dir = "duplicate_SCimilarity"
    n_cols = 128
else:
    print ("embedding method not supported yet")
    sys.exit(1)
os.makedirs(results_dir, exist_ok=True)

select_datasets = []
if args.select_datasets_file:
    with open(args.select_datasets_file, 'r') as f:
        select_datasets = [line.strip().split("\t")[0] for line in f if line.strip() and not line.startswith("#")]

obs_outfile = "obs.tsv.gz"

ucsc_file = "datasetLists/ucsc_datasets.txt"
cgCensus_non_primary_file ="datasetLists/cgCensus_non_primary_datasets.txt"
ignore_file = "datasetLists/ignore_datasets.txt"

if os.path.isdir(dup_dir):
    with open(ignore_file) as f:
        ignore_ids = {line.strip().split("\t")[0] for line in f
                      if line.strip() and not line.startswith("#")}
    dup_ids = {f.removesuffix(f"_{embedding}.npy")
               for f in os.listdir(dup_dir) if f.endswith(".npy")}
    missing_from_ignore = dup_ids - ignore_ids
    if missing_from_ignore:
        print(f"ERROR: {dup_dir} datasets not in ignore_datasets.txt:")
        for mid in sorted(missing_from_ignore):
            print(f"  {mid}")
        sys.exit(1)

with open(ucsc_file) as f:
    ucsc_ids = {line.strip().split("\t")[0] for line in f
                if line.strip() and not line.startswith("#")}
for check_dir in set(data_dirs + [dup_dir]):
    if not os.path.isdir(check_dir):
        continue
    ucsc_file_ids = {f.removesuffix(f"_{embedding}.npy")
                     for f in os.listdir(check_dir)
                     if f.endswith(".npy") and f.startswith("ucsc-")}
    missing_from_ucsc = ucsc_file_ids - ucsc_ids
    if missing_from_ucsc:
        print(f"ERROR: {check_dir} ucsc datasets not in ucsc_datasets.txt:")
        for mid in sorted(missing_from_ucsc):
            print(f"  {mid}")
        sys.exit(1)

dataset_list = []
cellCount = 0
remapfile_exist_datasets = []             # (real_dataset_id, dataset_id, mapped_filepath, doc_filepath)
                                          #   remap file (_remap.tsv + _obs_mapped.tsv.gz) found — UCSC mapped obs merged in as-is
arr_dtype = np.dtype('float32')
counter = 0
select_cols = ['disease', 'disease_ontology_term_id',
               'tissue', 'tissue_ontology_term_id',
               'cell_type', 'cell_type_ontology_term_id',
               'assay', 'assay_ontology_term_id',
               'development_stage', 'development_stage_ontology_term_id',
               "organism","organism_ontology_term_id",
               "sex","sex_ontology_term_id",
               "self_reported_ethnicity","self_reported_ethnicity_ontology_term_id",
               'donor_id',
               'sample_id',
               'suspension_type',
               'tissue_type',
               'dataset_id',
               'is_primary_data',
               ]

def get_ucsc_dataset_ids():
    datasets =[]
    # Read dataset IDs from the text file (one per line)
    with open(ucsc_file, "r") as f:
        dataset_info = [line.strip().split("\t") for line in f if (line.strip() and line[0] != "#")]
    for item in dataset_info:
        datasets.append(item[0])
    return datasets

def get_cgCensus_non_primary_dataset_ids():
    datasets =[]
    # Read dataset IDs from the text file (one per line)
    with open(cgCensus_non_primary_file, "r") as f:
        dataset_info = [line.strip().split("\t") for line in f if (line.strip() and line[0] != "#")]
    for item in dataset_info:
        datasets.append(item[0])
    return datasets

def get_ignore_dataset_ids():
    datasets =[]
    # Read dataset IDs from the text file (one per line)
    with open(ignore_file, "r") as f:
        dataset_info = [line.strip().split("\t") for line in f if (line.strip() and line[0] != "#")]
    for item in dataset_info:
        datasets.append(item[0])
    return datasets


def prescan_cell_count(data_dirs, embedding, select_datasets, ignore_datasets,
                       ucsc_datasets, cgCensus_non_primary_datasets):
    """Read .npy shapes + obs is_primary_data to estimate total cells before allocation."""
    total = 0
    seen = set()
    for singleDir in data_dirs:
        if not os.path.isdir(singleDir):
            continue
        if len(select_datasets) > 0:
            files = [f"{ds_id}_{embedding}.npy" for ds_id in select_datasets]
        else:
            _, _, walked = next(os.walk(singleDir))
            files = sorted(f for f in walked if f.endswith('.npy'))
        for file in files:
            filepath = os.path.join(singleDir, file)
            if not os.path.exists(filepath):
                continue
            dataset_id = file.removesuffix("_uce.npy").removesuffix("_scimilarity.npy")
            match = re.match(r'^((?:ucsc-)?[a-f0-9\-]+)_', file)
            real_dataset_id = match.group(1) if match else dataset_id
            if len(select_datasets) == 0 and real_dataset_id in ignore_datasets:
                continue
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            arr = np.load(filepath, mmap_mode='r')
            n = arr.shape[0]
            obs_filepath = os.path.join(singleDir, dataset_id + "_obs.tsv")
            if not os.path.exists(obs_filepath):
                obs_filepath = os.path.join(singleDir, dataset_id + "_obs.tsv.gz")
            if os.path.exists(obs_filepath) and \
               dataset_id not in ucsc_datasets and \
               dataset_id not in cgCensus_non_primary_datasets:
                try:
                    obs_cols = pd.read_csv(obs_filepath, sep='\t', nrows=0).columns.tolist()
                    if "is_primary_data" in obs_cols:
                        obs_primary = pd.read_csv(obs_filepath, sep='\t',
                                                  usecols=["is_primary_data"], low_memory=False)
                        n = int((obs_primary["is_primary_data"] == True).sum())
                except Exception:
                    pass
            total += n
    return total

def get_ucsc_collection():
    df = pd.read_csv(
        ucsc_file,
        sep="\t",
        comment="#",      # ignore any line starting with '#'
        index_col=0       # use the first column as index
    )
    return df

#cellxgene info
latest_census = cellxgene_census.open_soma(census_version = "latest")
cxgInfo = latest_census["census_info"]["datasets"].read().concat().to_pandas()

# fallback: Discovery API for datasets not in Census (e.g. MERFISH/spatial)
# keyed by CXG explorer URL ID (the UUID in explorer_url, which is used as the .npy filename)
# _discovery_index: cxg_id -> {collection_id, name}  (fetched once, no collection calls yet)
# _discovery_collection_cache: collection_id -> (doi, doi_label)  (fetched on demand)
def _build_discovery_index():
    import requests, re
    r = requests.get("https://api.cellxgene.cziscience.com/dp/v1/datasets/index")
    r.raise_for_status()
    index = {}
    for ds in r.json():
        url = ds.get("explorer_url", "")
        m = re.search(r'/e/([a-f0-9\-]+)\.cxg', url)
        if m:
            index[m.group(1)] = {"collection_id": ds.get("collection_id", ""), "name": ds.get("name", "")}
    return index

_discovery_collection_cache = {}

def get_discovery_info(cxg_id):
    import requests, re
    if cxg_id not in _discovery_index:
        return None
    ds = _discovery_index[cxg_id]
    coll_id = ds["collection_id"]
    if coll_id not in _discovery_collection_cache:
        rc = requests.get(f"https://api.cellxgene.cziscience.com/dp/v1/collections/{coll_id}")
        if rc.status_code == 200:
            coll = rc.json()
            doi_link = next((l["link_url"] for l in coll.get("links", []) if l.get("link_type") == "DOI"), "")
            doi = re.sub(r'^https?://doi\.org/', '', doi_link)
            _discovery_collection_cache[coll_id] = (doi, coll.get("summary_citation", ""))
        else:
            _discovery_collection_cache[coll_id] = ("", "")
    doi, doi_label = _discovery_collection_cache[coll_id]
    return {"collection_doi": doi, "collection_doi_label": doi_label, "dataset_title": ds["name"]}

print("Fetching Discovery API dataset index...")
_discovery_index = _build_discovery_index()
print(f"  {len(_discovery_index)} datasets indexed from Discovery")

#ucsc info
ucscInfo = get_ucsc_collection()

# file sets
ucsc_datasets = get_ucsc_dataset_ids()
cgCensus_non_primary_datasets = get_cgCensus_non_primary_dataset_ids()
ignore_datasets = get_ignore_dataset_ids()

if not obs_only:
    if shuffle:
        print("Pre-scanning datasets for total cell count...")
        prescan_total = prescan_cell_count(data_dirs, embedding, select_datasets, ignore_datasets,
                                           ucsc_datasets, cgCensus_non_primary_datasets)
        print(f"  Pre-scan estimate: {prescan_total} cells")
        nvme_tmp = os.path.join(nvme_dir, f"{embedding}_shuffle_{os.path.basename(os.path.normpath(results_dir))}.dat")
        print(f"  Creating NVMe memmap: {nvme_tmp}")
        uce_mmap = np.memmap(nvme_tmp, dtype=arr_dtype, mode='w+', shape=(prescan_total, n_cols))
        uce_write_pos = 0
    else:
        _uce_nvme_tmp = os.path.join(nvme_dir, f"{embedding}_{os.path.basename(os.path.normpath(results_dir))}.h5")
        print(f"  Writing UCE h5 to NVMe staging: {_uce_nvme_tmp}")
        fuce = h5py.File(_uce_nvme_tmp, "w")
        dset_uce = fuce.create_dataset("data",
                                       shape=(0, n_cols),
                                       maxshape=(None, n_cols),
                                       dtype=arr_dtype
                                       )

CHUNK = 200_000
if not uce_only:
    obs_outpath = os.path.join(results_dir, obs_outfile)
    if shuffle:
        import pyarrow as pa, pyarrow.parquet as pq
        _obs_tmp = os.path.join(nvme_dir, f"obs_tmp_{os.path.basename(os.path.normpath(results_dir))}.parquet")
        _pq_writer = None
    else:
        import gzip as _gzip
        _obs_nvme_tmp = os.path.join(nvme_dir, f"obs_{os.path.basename(os.path.normpath(results_dir))}.tsv.gz")
        print(f"  Writing obs to NVMe staging: {_obs_nvme_tmp}")
        _obs_out = _gzip.open(_obs_nvme_tmp, 'wt')
        _obs_header_written = False

# pre-scan obs column headers for select_datasets mode — schemas differ across datasets
union_obs_cols = None
if len(select_datasets) > 0 and not uce_only:
    col_order: dict = {}
    for ds_id in select_datasets:
        for sdir in data_dirs:
            obs_fp = os.path.join(sdir, ds_id + "_obs.tsv")
            if not os.path.exists(obs_fp):
                obs_fp = os.path.join(sdir, ds_id + "_obs.tsv.gz")
            if not os.path.exists(obs_fp):
                continue
            cols = pd.read_csv(obs_fp, sep='\t', nrows=0).columns.tolist()
            cols = ["old_index" if c == "Unnamed: 0" else c for c in cols]
            remap_fp  = os.path.join(sdir, ds_id + "_remap.tsv")
            mapped_fp = os.path.join(sdir, ds_id + "_obs_mapped.tsv.gz")
            doc_fp    = os.path.join(sdir, ds_id + "_obs_mapped_doc.txt")
            if not os.path.exists(doc_fp):
                doc_fp = os.path.join(sdir, ds_id + ".report.md")
            if os.path.exists(remap_fp) and os.path.exists(mapped_fp) and os.path.exists(doc_fp):
                mapped_cols = pd.read_csv(mapped_fp, sep='\t', nrows=0).columns.tolist()
                cols += [c for c in mapped_cols if c not in cols]
            for c in cols:
                col_order[c] = None
            break
    union_obs_cols = [c for c in col_order if c != "new_index"]
    for c in ["dataset_id", "collection_doi", "collection_doi_label", "dataset_title"]:
        if c not in union_obs_cols:
            union_obs_cols.append(c)
    print(f"  Union obs schema: {len(union_obs_cols)} columns across {len(select_datasets)} select_datasets")

# Pre-declare parquet schema for shuffle mode so every chunk uses the same types.
# new_index is reset into a regular column before each write, so include it here.
# All columns declared as string to avoid dtype inference differences across chunks
# (e.g. reindex fills missing cols with np.nan which PyArrow infers as double).
_pq_schema_arrow = None
if shuffle and not uce_only:
    if union_obs_cols is not None:
        _pq_schema_arrow = pa.schema(
            [('new_index', pa.string())] + [(col, pa.string()) for col in union_obs_cols]
        )

for singleDir in data_dirs:
    # When select_datasets is provided, iterate in its order to guarantee deterministic row ordering.
    # When combining all datasets, walk the filesystem (order is non-deterministic — use --shuffle for randomization).
    if len(select_datasets) > 0:
        all_files = [f"{ds_id}_{embedding}.npy" for ds_id in select_datasets]
    else:
        _, _, walked_files = next(os.walk(singleDir))
        all_files = sorted(f for f in walked_files if f.endswith('.npy'))

    for file in all_files:
        if not os.path.exists(os.path.join(singleDir, file)):
            if singleDir != dup_dir:
                print(f"  WARNING: {file} not found in {singleDir}, skipping")
            continue
        dataset_id = file.removesuffix("_uce.npy").removesuffix("_scimilarity.npy")
        match = re.match(r'^((?:ucsc-)?[a-f0-9\-]+)_', file)
        real_dataset_id = match.group(1) if match else dataset_id

        if len(select_datasets) == 0 and real_dataset_id in ignore_datasets:
            continue

        print(singleDir, dataset_id)
        if dataset_id not in dataset_list:
            dataset_list.append(dataset_id)

            uce_filepath = os.path.join(singleDir, file)
            uce_raw = np.load(uce_filepath, mmap_mode='r')
            obs_filepath = os.path.join(singleDir, dataset_id + "_obs.tsv")
            if not os.path.exists(obs_filepath):
                obs_filepath = os.path.join(singleDir, dataset_id + "_obs.tsv.gz")

            # remap: detect new columns and set up a chunked reader aligned with obs/uce
            remap_filepath  = os.path.join(singleDir, dataset_id + "_remap.tsv")
            mapped_filepath = os.path.join(singleDir, dataset_id + "_obs_mapped.tsv.gz")
            doc_filepath    = os.path.join(singleDir, dataset_id + "_obs_mapped_doc.txt")
            if not os.path.exists(doc_filepath):
                doc_filepath = os.path.join(singleDir, dataset_id + ".report.md")
            has_remap = (os.path.exists(remap_filepath) and
                         os.path.exists(mapped_filepath) and
                         os.path.exists(doc_filepath))
            mapped_reader  = None
            remap_new_cols = []
            if has_remap:
                _obs_hdr    = pd.read_csv(obs_filepath,    sep='\t', nrows=0).columns.tolist()
                _mapped_hdr = pd.read_csv(mapped_filepath, sep='\t', nrows=0).columns.tolist()
                remap_new_cols = [c for c in _mapped_hdr if c not in _obs_hdr]
                if remap_new_cols:
                    # dtype=str forces string typing without disabling NA detection:
                    # pandas still substitutes NaN for cells matching its default NA
                    # tokens ('', 'NA', 'null', 'None', etc.) before typing survivors
                    # as str, matching the pre-chunking code's missing-value semantics.
                    mapped_reader = pd.read_csv(mapped_filepath, sep='\t',
                                                usecols=remap_new_cols,
                                                dtype=str, low_memory=False, chunksize=CHUNK)
                print(f"  Merging UCSC remap for {dataset_id} from {mapped_filepath}")
                remapfile_exist_datasets.append((real_dataset_id, dataset_id, mapped_filepath, doc_filepath))

            # fetch collection info once per dataset; stamped onto every chunk
            if not uce_only:
                if dataset_id not in ucsc_datasets:
                    cxg_row = cxgInfo[cxgInfo.dataset_id == dataset_id]
                    if len(cxg_row) > 0:
                        _doi_label = cxg_row["collection_doi_label"].values[0]
                        _doi       = cxg_row["collection_doi"].values[0]
                        _title     = cxg_row["dataset_title"].values[0]
                    else:
                        disc = get_discovery_info(dataset_id)
                        if disc is not None:
                            _doi_label = disc["collection_doi_label"]
                            _doi       = disc["collection_doi"]
                            _title     = disc["dataset_title"]
                        else:
                            print(f"  WARNING: {dataset_id} not found in Census or Discovery; collection fields set to NaN")
                            _doi_label = np.nan
                            _doi       = np.nan
                            _title     = np.nan
                else:
                    _doi       = ucscInfo.at[dataset_id, "collection_doi"]
                    _doi_label = ucscInfo.at[dataset_id, "collection_doi_label"]
                    _title     = ucscInfo.at[dataset_id, "dataset_title"]

            # dtype=str forces string typing without disabling NA detection: pandas
            # still substitutes NaN for cells matching its default NA tokens ('',
            # 'NA', 'null', 'None', etc.) before typing survivors as str, matching
            # the pre-chunking code's missing-value semantics.
            obs_reader = pd.read_csv(obs_filepath, sep='\t',
                                     dtype=str, low_memory=False, chunksize=CHUNK)
            chunk_iter = _zip_strict_chunks(uce_raw.shape[0], CHUNK, obs_reader, mapped_reader, dataset_id)

            any_written = False
            for uce_start, obs_chunk, mapped_chunk in chunk_iter:
                uce_end   = min(uce_start + CHUNK, uce_raw.shape[0])
                uce_chunk = np.array(uce_raw[uce_start:uce_end])
                _madvise_dontneed(uce_raw, uce_start, uce_end)

                mask_uce = ~np.isnan(uce_chunk).any(axis=1)
                if "is_primary_data" in obs_chunk.columns and \
                   dataset_id not in ucsc_datasets and \
                   dataset_id not in cgCensus_non_primary_datasets:
                    mask_primary = (obs_chunk["is_primary_data"].str.strip().str.lower() == 'true').values
                else:
                    mask_primary = np.ones(len(uce_chunk), dtype=bool)

                combined_mask = mask_uce & mask_primary
                if combined_mask.sum() == 0:
                    continue

                uce_filtered = uce_chunk[combined_mask]
                cellCount += uce_filtered.shape[0]
                any_written = True
                n_chunks = -(-uce_raw.shape[0] // CHUNK)
                print(f"  chunk {uce_start//CHUNK + 1}/{n_chunks}: {uce_filtered.shape[0]:,} cells, cumulative {cellCount:,}")

                if not uce_only:
                    obs_filtered = obs_chunk[combined_mask].copy()
                    if mapped_chunk is not None:
                        for col in remap_new_cols:
                            obs_filtered[col] = mapped_chunk[col].values[combined_mask]
                    obs_filtered['new_index'] = real_dataset_id + "_" + obs_filtered.index.astype(str)
                    obs_filtered.rename(columns={"Unnamed: 0": "old_index"}, inplace=True)
                    obs_filtered.set_index('new_index', inplace=True)
                    obs_filtered = obs_filtered.assign(dataset_id=real_dataset_id)
                    if len(select_datasets) == 0:
                        cols_to_use = [c for c in select_cols if c in obs_filtered.columns]
                        obs_filtered = obs_filtered[cols_to_use]
                    obs_filtered["collection_doi"]       = _doi
                    obs_filtered["collection_doi_label"] = _doi_label
                    obs_filtered["dataset_title"]        = _title
                    if union_obs_cols is not None:
                        obs_filtered = obs_filtered.reindex(columns=union_obs_cols)
                    if shuffle:
                        # Reset index so new_index becomes a regular column in the parquet.
                        # Using an explicit all-string schema avoids dtype inference issues:
                        # reindex fills missing cols with np.nan which PyArrow otherwise
                        # infers as double instead of string, causing schema mismatches.
                        obs_for_pq = obs_filtered.reset_index()
                        if _pq_schema_arrow is not None:
                            table = pa.Table.from_pandas(obs_for_pq,
                                                         schema=_pq_schema_arrow,
                                                         preserve_index=False)
                        else:
                            table = pa.Table.from_pandas(obs_for_pq,
                                                         preserve_index=False)
                        if _pq_writer is None:
                            _pq_writer = pq.ParquetWriter(_obs_tmp, table.schema)
                        _pq_writer.write_table(table)
                    else:
                        obs_filtered.to_csv(_obs_out, sep='\t', header=not _obs_header_written)
                        _obs_header_written = True

                if not obs_only:
                    if shuffle:
                        uce_mmap[uce_write_pos:uce_write_pos + uce_filtered.shape[0]] = uce_filtered
                        uce_write_pos += uce_filtered.shape[0]
                    else:
                        dset_uce.resize(dset_uce.shape[0] + uce_filtered.shape[0], axis=0)
                        dset_uce[-uce_filtered.shape[0]:] = uce_filtered

            del uce_raw
            if any_written:
                counter += 1
                print(counter, "total cell count:", cellCount)
            else:
                print("All embedding data are NaN (very rare) or is_primary_data are all False (more likely)")
        else:
            print ("Duplicate dataset")
if not obs_only and not shuffle:
    fuce.close()
    uce_efs_path = os.path.join(results_dir, embedding_outfile)
    print(f"Moving UCE h5 NVMe -> EFS: {_uce_nvme_tmp} -> {uce_efs_path}")
    shutil.move(_uce_nvme_tmp, uce_efs_path)

if shuffle and not obs_only:
    actual_total = uce_write_pos
    assert actual_total == prescan_total, f"Cell count mismatch: prescan={prescan_total}, actual={actual_total}"
    print(f"Generating shuffle index ({actual_total} cells)...")
    shuffle_idx = np.random.permutation(actual_total)
    shuffle_idx_path = os.path.join(results_dir, "shuffle_index.uce.npy")
    np.save(shuffle_idx_path, shuffle_idx)
    print(f"Shuffle index saved to {shuffle_idx_path}")
    print("Writing shuffled UCE to HDF5 (NVMe -> EFS)...")
    uce_h5_path = os.path.join(results_dir, embedding_outfile)
    shuffle_chunk = 100_000
    with h5py.File(uce_h5_path, 'w') as fout:
        dset_out = fout.create_dataset('data', shape=(actual_total, n_cols), dtype=arr_dtype)
        for out_start in range(0, actual_total, shuffle_chunk):
            out_end = min(out_start + shuffle_chunk, actual_total)
            dset_out[out_start:out_end] = uce_mmap[shuffle_idx[out_start:out_end]]
    del uce_mmap
    os.remove(nvme_tmp)
    print("UCE shuffle complete.")

if not uce_only:
    if shuffle:
        if _pq_writer is not None:
            _pq_writer.close()
        print("Reading obs parquet for shuffle reorder...")
        combined_obs = pd.read_parquet(_obs_tmp)
        print("Shuffling obs...")
        combined_obs = combined_obs.iloc[shuffle_idx].reset_index(drop=True)
        if 'new_index' in combined_obs.columns:
            combined_obs = combined_obs.set_index('new_index')
        combined_obs.to_csv(obs_outpath, sep='\t', compression='gzip')
        os.remove(_obs_tmp)
    else:
        _obs_out.close()
        print(f"Moving obs NVMe -> EFS: {_obs_nvme_tmp} -> {obs_outpath}")
        shutil.move(_obs_nvme_tmp, obs_outpath)
    print(f"Obs written to {obs_outpath}")

print(f"these {len(dataset_list)} datasets are combined")
print (dataset_list)

# Write wrangling notes
from datetime import datetime

def _dataset_info(dataset_id, ucsc_datasets, cxgInfo, ucscInfo):
    """Return (collection_doi_label, dataset_title) for a dataset_id."""
    if dataset_id in ucsc_datasets:
        try:
            return ucscInfo.at[dataset_id, "collection_doi_label"], ucscInfo.at[dataset_id, "dataset_title"]
        except KeyError:
            return "—", "—"
    else:
        row = cxgInfo[cxgInfo.dataset_id == dataset_id]
        if len(row) > 0:
            return row["collection_doi_label"].values[0], row["dataset_title"].values[0]
        return "—", "—"

wrangling_notes_path = os.path.join(results_dir, "wrangling_notes_combineEmbedding.txt")
with open(wrangling_notes_path, "w") as wf:
    wf.write(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    wf.write(f"embedding: {embedding}\n")
    wf.write(f"results_dir: {results_dir}\n")
    wf.write("\n")
    wf.write("Combine .obs fields from source embedding files\n")
    wf.write("Add column 'collection_doi', populate with publication doi from source data\n")
    wf.write("Add column 'collection_doi_label', populate with publication doi label from source data\n")
    wf.write("Add column 'dataset_title', populate with dataset title from source data\n")
    wf.write(f"\nDatasets processed: {len(dataset_list)}\n")
    if remapfile_exist_datasets:
        wf.write("UCSC remap (_remap.tsv + _obs_mapped.tsv.gz): mapped columns merged into obs as-is (no renaming)\n")
        wf.write(f"\nUCSC remap merged ({len(remapfile_exist_datasets)} datasets):\n")
        for real_ds_id, ds_id, mapped_fp, doc_fp in remapfile_exist_datasets:
            doi, title = _dataset_info(ds_id, ucsc_datasets, cxgInfo, ucscInfo)
            wf.write(f"\n  {real_ds_id}  '{title}'  {doi}\n")
            if os.path.exists(doc_fp):
                with open(doc_fp) as doc_f:
                    for line in doc_f:
                        wf.write("    " + line)
                wf.write("\n")
print(f"Wrangling notes written to {wrangling_notes_path}")
