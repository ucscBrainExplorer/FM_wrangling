import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--embedding",
    required=True,
    choices=["uce", "uce_brain_512", "scimilarity"],
    help="Embedding type to use (uce, uce_brain_512 or scimilarity)"
)

parser.add_argument(
    "--results_dir",
    required=True,
    help="Directory path where the output results will be saved"
)

parser.add_argument(
    "--select-datasets-file",
    default=None,
    help="File of datasets to process, one per line: dataset ID, then optionally <tab> source h5ad URL "
         "(else taken from <id>_source.txt); the URLs are written to <results_dir>/dataset_ids.tsv"
)
parser.add_argument(
    "--obs-columns-file",
    default=None,
    help="With --select-datasets-file only: file listing the obs columns to output, one per "
         "line ('#' comments allowed), in output order, instead of the union of all columns"
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
    "--staging-dir",
    default="/nvme",
    help="Fast local scratch for staging output files before moving them to results_dir (default: /nvme)"
)

args = parser.parse_args()

import os, sys, re, glob

if args.shuffle and (args.obs_only or args.uce_only):
    print("ERROR: --shuffle cannot be used with --obs-only or --uce-only — obs and uce must be written together to stay in sync.")
    sys.exit(1)

if args.obs_columns_file and not args.select_datasets_file:
    print("ERROR: --obs-columns-file requires --select-datasets-file (a full run always writes the fixed standard column set).")
    sys.exit(1)

shuffle = args.shuffle
embedding = args.embedding
results_dir = args.results_dir
staging_dir = args.staging_dir
import shutil
import mmap
import numpy as np
import pandas as pd
import h5py
from obs_shuffle import write_shuffled_obs   # FM_wrangling/obs_shuffle.py (script dir is on sys.path)

# After reading a mmap chunk, tell the kernel it can reclaim those pages, keeping
# RSS flat while sequentially scanning a large read-only mmap. Uses mmap.mmap's
# native madvise() (stdlib since Python 3.8, Unix only) via numpy.memmap's
# underlying mmap object (its private `._mmap` attribute — undocumented, but
# stable across numpy versions in practice) instead of hand-rolled ctypes/libc
# calls. This is purely an optimization hint to the OS, not a correctness
# requirement, so any failure to apply it (non-Unix, an unexpected numpy
# internal, etc.) is caught and downgrades to a one-time warning instead of
# crashing the run.
_PAGE_SIZE = mmap.PAGESIZE
_madvise_available = True

def _madvise_dontneed(memmap_arr, start_row, end_row):
    global _madvise_available
    if not _madvise_available:
        return
    try:
        mm     = memmap_arr._mmap
        stride = memmap_arr.strides[0]
        start_byte = memmap_arr.offset + start_row * stride
        end_byte   = memmap_arr.offset + end_row   * stride
        pg_start = (start_byte // _PAGE_SIZE) * _PAGE_SIZE
        pg_end   = -(-end_byte // _PAGE_SIZE) * _PAGE_SIZE
        mm.madvise(mmap.MADV_DONTNEED, pg_start, pg_end - pg_start)
    except (AttributeError, OSError) as e:
        print(f"  WARNING: madvise(MADV_DONTNEED) unavailable ({e}); continuing "
              f"without page-cache release (RSS may grow more than expected)")
        _madvise_available = False


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
elif embedding == "uce_brain_512":
    embedding_outfile = "uce.h5"
    data_dirs =["UCE_brain_512_npy_obs", "duplicate_UCE_brain_512"]
    dup_dir = "duplicate_UCE_brain_512"
    n_cols = 512
    embedding = "uce"  # input files named *_uce.npy like uce; location and width differ
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
        select_rows = [line.rstrip("\n").split("\t") for line in f if line.strip() and not line.startswith("#")]
    select_datasets = [row[0].strip() for row in select_rows]
    if not select_datasets:
        print(f"ERROR: {args.select_datasets_file} lists no dataset ids")
        sys.exit(1)

    # Source h5ad URL per dataset: column 2 of the select file, else <id>_source.txt written by
    # getEmbeddingfromS3.py. Both must agree, every dataset must have one, and the result is
    # written to <results_dir>/dataset_ids.tsv as the atlas's record of where its data came from.
    listed = {row[0].strip(): row[1].strip() for row in select_rows if len(row) > 1 and row[1].strip()}
    source_urls, problems = {}, []
    for ds_id in select_datasets:
        recorded = None
        for sdir in data_dirs:
            src_path = os.path.join(sdir, f"{ds_id}_source.txt")
            if os.path.exists(src_path):
                with open(src_path) as f:
                    recorded = f.read().strip()
                if recorded == "unknown":
                    recorded = None
                break
        if listed.get(ds_id) and recorded and listed[ds_id] != recorded:
            problems.append(f"{ds_id}: select file says {listed[ds_id]}, {src_path} says {recorded}")
        elif not (listed.get(ds_id) or recorded):
            problems.append(f"{ds_id}: no source URL in {args.select_datasets_file} or <id>_source.txt")
        else:
            source_urls[ds_id] = listed.get(ds_id) or recorded
    if problems:
        print("ERROR: source h5ad URL missing or conflicting (add it as column 2 of the select file):")
        for msg in problems:
            print(f"  {msg}")
        sys.exit(1)
    dataset_ids_out = os.path.join(results_dir, "dataset_ids.tsv")
    with open(dataset_ids_out, "w") as f:
        for ds_id in select_datasets:
            f.write(f"{ds_id}\t{source_urls[ds_id]}\n")
    print(f"Source URLs for {len(select_datasets)} datasets written to {dataset_ids_out}")

obs_outfile = "obs.tsv.gz"

ucsc_file = "datasetLists/ucsc_datasets.txt"
cgCensus_non_primary_file ="datasetLists/cgCensus_non_primary_datasets.txt"
ignore_file = "datasetLists/ignore_datasets.txt"
manual_column_file = "datasetLists/manual_column.json"

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
dataset_collection_info = []              # (real_dataset_id, collection_doi, collection_doi_label, dataset_title)
                                          #   for the wrangling notes — actual values, not a generic description
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

#cellxgene info: newest datasetLists/datasets.info.cxg.YYYY-MM-DD.txt (from fetch_cxg_datasets.py);
# replaces cellxgene_census.open_soma(), whose tiledbsoma needs AVX (courtyard has none)
_cxg_info_files = sorted(glob.glob("datasetLists/datasets.info.cxg.*.txt"))
if not _cxg_info_files:
    print("ERROR: no datasetLists/datasets.info.cxg.*.txt found; run datasetLists/fetch_cxg_datasets.py")
    sys.exit(1)
cxg_info_file = _cxg_info_files[-1]
print(f"  CELLxGENE dataset info: {cxg_info_file}")
cxgInfo = pd.read_csv(cxg_info_file, sep="\t", dtype=str).rename(columns={"#dataset_id": "dataset_id"})

# fallback: Discovery API for datasets not in the datasets.info table (e.g. MERFISH/spatial)
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
        if prescan_total == 0:
            # Fail fast here instead of running the full pipeline (Discovery API fetch,
            # an empty shuffle, an empty uce.h5 write) only to crash confusingly at the
            # very end trying to read a Parquet obs file that was never written because
            # there was nothing to write. 0 cells almost always means a dataset_id in
            # --select-datasets-file doesn't match any actual "<id>_{embedding}.npy"
            # filename (e.g. a missing "_AIT" suffix) — check that first.
            print(f"ERROR: pre-scan found 0 cells - nothing would be written. Aborting.")
            if len(select_datasets) > 0:
                print(f"  select_datasets ({len(select_datasets)}): {select_datasets}")
                print(f"  Check each id matches an actual '<id>_{embedding}.npy' file "
                     f"in one of: {data_dirs}")
            else:
                print(f"  No .npy files matched in {data_dirs} after ignore_datasets filtering")
            sys.exit(1)
        staging_tmp = os.path.join(staging_dir, f"{embedding}_shuffle_{os.path.basename(os.path.normpath(results_dir))}.dat")
        print(f"  Creating staging memmap: {staging_tmp}")
        uce_mmap = np.memmap(staging_tmp, dtype=arr_dtype, mode='w+', shape=(prescan_total, n_cols))
        uce_write_pos = 0
    else:
        _uce_staging_tmp = os.path.join(staging_dir, f"{embedding}_{os.path.basename(os.path.normpath(results_dir))}.h5")
        print(f"  Writing UCE h5 to staging: {_uce_staging_tmp}")
        fuce = h5py.File(_uce_staging_tmp, "w")
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
        _obs_tmp = os.path.join(staging_dir, f"obs_tmp_{os.path.basename(os.path.normpath(results_dir))}.parquet")
        _pq_writer = None
    else:
        import gzip as _gzip
        _obs_staging_tmp = os.path.join(staging_dir, f"obs_{os.path.basename(os.path.normpath(results_dir))}.tsv.gz")
        print(f"  Writing obs to staging: {_obs_staging_tmp}")
        _obs_out = _gzip.open(_obs_staging_tmp, 'wt')
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

# Fixed target obs column set, used for both output paths (TSV and shuffle's Parquet
# staging) and both dataset-selection modes, so every chunk from every dataset ends
# up with the identical column set — a dataset missing one of these gets it padded
# with null rather than silently dropping the column (which would misalign a plain
# TSV's header against later rows, or crash Parquet's schema-locked writer).
if union_obs_cols is not None:
    target_obs_cols = union_obs_cols          # select_datasets mode: union of actual headers
else:
    target_obs_cols = select_cols + ['collection_doi', 'collection_doi_label', 'dataset_title']

# --obs-columns-file (select_datasets mode only): explicit output column list, in
# file order, replacing the union above. Columns a rule or the reindex needs as a source (e.g. Subclass,
# load_id) are read from each dataset's obs before the reindex, so they need not
# be listed; only the output is restricted.
obs_columns_file = args.obs_columns_file
if obs_columns_file and not uce_only:
    with open(obs_columns_file) as f:
        _listed = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    _dups = sorted({c for c in _listed if _listed.count(c) > 1})
    if _dups or "new_index" in _listed:
        print(f"ERROR: {obs_columns_file}: duplicate columns {_dups} or 'new_index' "
              f"(the index, always written) listed")
        sys.exit(1)
    _absent = [c for c in _listed if c not in (union_obs_cols or [])]
    if _absent:
        print(f"  WARNING: {obs_columns_file}: not in any selected dataset's obs "
              f"(empty unless a manual_column.json rule fills them): {_absent}")
    target_obs_cols = _listed
    print(f"  obs columns: {len(target_obs_cols)} from {obs_columns_file}")

# manual_column.json overrides, applied in both modes per chunk before the reindex,
# while each dataset's own columns (e.g. supercluster_term, Group) are still present.
# Full run: a target outside the fixed target_obs_cols is warned about and ignored.
# select_datasets mode: a target missing from the union schema is appended to it
# (only for selected datasets), and source columns stay in the output; with
# --obs-columns-file, only listed targets are applied and listed columns output.
#   {"<dataset_id>": {"<target col>": {"type": "static"|"column"|"map", "value": ...}, ...}}
#   static: value is the literal; column: value is a source column copied as-is;
#   map: value is a source column whose values are translated through "map"
#        ({"<source value>": "<new value>" | null, ...}); null maps to missing (NaN);
#        unmapped values pass through unchanged and are reported.
#   Any rule may carry an optional "note" string, copied into the wrangling notes.
# Non-dict entries (collection_doi_label, dataset_title) are informational only.
manual_rules = {}                         # real_dataset_id -> [(target, type, value, mapping, note)]
manual_rule_notes = []                    # lines for the wrangling notes
manual_map_unmapped = {}                  # (real_dataset_id, target, source) -> set of unmapped values
if not uce_only:
    import json
    # real ids of the selected datasets, derived the same way as real_dataset_id below
    selected_real_ids = set()
    for _sid in select_datasets:
        _m = re.match(r'^((?:ucsc-)?[a-f0-9\-]+)_', f"{_sid}_{embedding}.npy")
        selected_real_ids.add(_m.group(1) if _m else _sid)
    with open(manual_column_file) as f:
        for ds_id, cfg in json.load(f).items():
            for target, rule in cfg.items():
                if not isinstance(rule, dict):
                    continue
                if rule.get("type") not in ("static", "column", "map") or "value" not in rule:
                    print(f"ERROR: {manual_column_file}: {ds_id}.{target} needs \"type\": "
                          f"\"static\"|\"column\"|\"map\" and a \"value\" (got {rule!r})")
                    sys.exit(1)
                if rule["type"] == "map" and not (isinstance(rule.get("map"), dict) and rule["map"]):
                    print(f"ERROR: {manual_column_file}: {ds_id}.{target} \"type\": \"map\" needs a "
                          f"non-empty \"map\" object (got {rule!r})")
                    sys.exit(1)
                if union_obs_cols is not None and ds_id not in selected_real_ids:
                    continue
                if target not in target_obs_cols:
                    if union_obs_cols is None or obs_columns_file:
                        # would be dropped by the reindex anyway
                        print(f"  WARNING: {manual_column_file}: {ds_id}.{target} is not an output obs "
                              f"column; rule ignored")
                        continue
                    target_obs_cols.append(target)
                # str(): obs is all-string; a JSON number would break the shuffle
                # path's all-string Parquet schema
                mapping = ({str(k): (np.nan if v is None else str(v)) for k, v in rule["map"].items()}
                           if rule["type"] == "map" else None)
                manual_rules.setdefault(ds_id, []).append((target, rule["type"], str(rule["value"]), mapping,
                                                           rule.get("note")))
    print(f"  manual_column.json: {sum(map(len, manual_rules.values()))} rules for {len(manual_rules)} datasets")

# Pre-declare parquet schema for shuffle mode so every chunk uses the same types.
# new_index is reset into a regular column before each write, so include it here.
# All columns declared as string to avoid dtype inference differences across chunks
# (e.g. reindex fills missing cols with np.nan which PyArrow infers as double).
_pq_schema_arrow = None
if shuffle and not uce_only:
    _pq_schema_arrow = pa.schema(
        [('new_index', pa.string())] + [(col, pa.string()) for col in target_obs_cols]
    )

for singleDir in data_dirs:
    # When select_datasets is provided, iterate in its order to guarantee deterministic row ordering.
    # When combining all datasets, walk the filesystem in sorted filename order. This
    # order is deterministic given the same files and dataset lists, which is what lets
    # apply_shuffle_index.py re-apply a saved shuffle_index.uce.npy to an --obs-only rerun.
    if not os.path.isdir(singleDir):
        continue
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

            # remap: detect new columns and set up a chunked reader aligned with obs/uce.
            # Only in select_datasets mode (union_obs_cols set): a full run reindexes
            # obs onto the fixed select_cols set, which would drop the remap columns
            # anyway, so they are neither read nor reported as merged.
            remap_filepath  = os.path.join(singleDir, dataset_id + "_remap.tsv")
            mapped_filepath = os.path.join(singleDir, dataset_id + "_obs_mapped.tsv.gz")
            doc_filepath    = os.path.join(singleDir, dataset_id + "_obs_mapped_doc.txt")
            if not os.path.exists(doc_filepath):
                doc_filepath = os.path.join(singleDir, dataset_id + ".report.md")
            has_remap = (union_obs_cols is not None and
                         os.path.exists(remap_filepath) and
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
                            print(f"  WARNING: {dataset_id} not found in {cxg_info_file} or Discovery; collection fields set to NaN")
                            _doi_label = np.nan
                            _doi       = np.nan
                            _title     = np.nan
                else:
                    _doi       = ucscInfo.at[dataset_id, "collection_doi"]
                    _doi_label = ucscInfo.at[dataset_id, "collection_doi_label"]
                    _title     = ucscInfo.at[dataset_id, "dataset_title"]
                dataset_collection_info.append((real_dataset_id, _doi, _doi_label, _title))

            # resolve this dataset's manual_column.json rules against its obs header;
            # a "column"/"map" rule whose source column is missing is skipped with a warning
            ds_rules = []
            if real_dataset_id in manual_rules:
                _hdr = set(pd.read_csv(obs_filepath, sep='\t', nrows=0).columns)
                for target, rtype, value, mapping, rnote in manual_rules[real_dataset_id]:
                    if rtype in ("column", "map") and value not in _hdr:
                        note = (f"SKIPPED map column '{value}' -> '{target}' for dataset "
                                f"'{real_dataset_id}' '{_doi_label}' '{_title}': source column not in obs")
                        print(f"  WARNING: {note}")
                    else:
                        ds_rules.append((target, rtype, value, mapping))
                        if rtype == "static":
                            note = f"Set '{target}' = '{value}'"
                        elif rtype == "column":
                            note = f"Map column '{value}' -> '{target}'"
                        else:
                            note = (f"Map column '{value}' -> '{target}' via value map "
                                    + ", ".join(f"'{k}'->" + ("NaN" if pd.isna(v) else f"'{v}'")
                                                for k, v in mapping.items()))
                        note += f" for dataset '{real_dataset_id}' '{_doi_label}' '{_title}'"
                        if rnote:
                            note += f" -- NOTE: {rnote}"
                        print(f"  {note}")
                    manual_rule_notes.append(note)

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
                    obs_filtered["collection_doi"]       = _doi
                    obs_filtered["collection_doi_label"] = _doi_label
                    obs_filtered["dataset_title"]        = _title
                    # manual_column.json overrides (see manual_rules)
                    for target, rtype, value, mapping in ds_rules:
                        if rtype == "static":
                            obs_filtered[target] = value
                        elif rtype == "column":
                            obs_filtered[target] = obs_filtered[value]
                        else:
                            src = obs_filtered[value]
                            hit = src.isin(mapping.keys())
                            unmapped = set(src[~hit & src.notna()].unique())
                            if unmapped:
                                manual_map_unmapped.setdefault((real_dataset_id, target, value), set()).update(unmapped)
                            obs_filtered[target] = src.map(mapping).where(hit, src)
                    # Pad/select onto the fixed target column set (see target_obs_cols
                    # above) so every dataset's chunks share the same columns, whether
                    # or not this particular dataset's obs.tsv has all of them.
                    obs_filtered = obs_filtered.reindex(columns=target_obs_cols)
                    if shuffle:
                        # Reset index so new_index becomes a regular column in the parquet.
                        # Using an explicit all-string schema avoids dtype inference issues:
                        # reindex fills missing cols with np.nan which PyArrow otherwise
                        # infers as double instead of string, causing schema mismatches.
                        # _pq_schema_arrow is always set here: it's built unconditionally
                        # whenever shuffle and not uce_only (see above), which is exactly
                        # this branch's own condition.
                        obs_for_pq = obs_filtered.reset_index()
                        table = pa.Table.from_pandas(obs_for_pq,
                                                     schema=_pq_schema_arrow,
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
    uce_out_path = os.path.join(results_dir, embedding_outfile)
    print(f"Moving UCE h5 staging -> results_dir: {_uce_staging_tmp} -> {uce_out_path}")
    shutil.move(_uce_staging_tmp, uce_out_path)

if shuffle and not obs_only:
    actual_total = uce_write_pos
    assert actual_total == prescan_total, f"Cell count mismatch: prescan={prescan_total}, actual={actual_total}"
    print(f"Generating shuffle index ({actual_total} cells)...")
    shuffle_idx = np.random.permutation(actual_total)
    shuffle_idx_path = os.path.join(results_dir, "shuffle_index.uce.npy")
    np.save(shuffle_idx_path, shuffle_idx)
    print(f"Shuffle index saved to {shuffle_idx_path}")
    print("Writing shuffled UCE to HDF5 (staging -> results_dir)...")
    uce_h5_path = os.path.join(results_dir, embedding_outfile)
    shuffle_chunk = 100_000
    with h5py.File(uce_h5_path, 'w') as fout:
        dset_out = fout.create_dataset('data', shape=(actual_total, n_cols), dtype=arr_dtype)
        for out_start in range(0, actual_total, shuffle_chunk):
            out_end = min(out_start + shuffle_chunk, actual_total)
            dset_out[out_start:out_end] = uce_mmap[shuffle_idx[out_start:out_end]]
    del uce_mmap
    os.remove(staging_tmp)
    print("UCE shuffle complete.")

if not uce_only:
    if shuffle:
        if _pq_writer is not None:
            _pq_writer.close()
        print("Writing shuffled obs in chunks (staging)...")
        _obs_shuf_staging_tmp = os.path.join(staging_dir, f"obs_{os.path.basename(os.path.normpath(results_dir))}.tsv.gz")
        write_shuffled_obs(_obs_tmp, shuffle_idx, _obs_shuf_staging_tmp)
        os.remove(_obs_tmp)
        print(f"Moving obs staging -> results_dir: {_obs_shuf_staging_tmp} -> {obs_outpath}")
        shutil.move(_obs_shuf_staging_tmp, obs_outpath)
    else:
        _obs_out.close()
        print(f"Moving obs staging -> results_dir: {_obs_staging_tmp} -> {obs_outpath}")
        shutil.move(_obs_staging_tmp, obs_outpath)
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
    wf.write(f"embedding: {args.embedding}\n")
    wf.write(f"results_dir: {results_dir}\n")
    if obs_columns_file:
        wf.write(f"obs columns: {len(target_obs_cols)} listed in {obs_columns_file}\n")
    wf.write("\n")
    wf.write("Combine .obs fields from source embedding files\n")
    wf.write("Added columns 'collection_doi', 'collection_doi_label', 'dataset_title' per dataset:\n")
    def _fmt(v):
        return "(none)" if (isinstance(v, float) and np.isnan(v)) or v == "" else str(v)
    for real_ds_id, doi, doi_label, title in dataset_collection_info:
        wf.write(f"  {real_ds_id}\n")
        wf.write(f"    collection_doi:       {_fmt(doi)}\n")
        wf.write(f"    collection_doi_label: {_fmt(doi_label)}\n")
        wf.write(f"    dataset_title:        {_fmt(title)}\n")
    wf.write(f"\nDatasets processed: {len(dataset_list)}\n")
    if manual_rule_notes:
        wf.write(f"\nmanual_column.json overrides applied in combine "
                 f"({'select_datasets' if union_obs_cols is not None else 'full run'}; "
                 f"{manual_column_file}):\n")
        for note in manual_rule_notes:
            wf.write(f"  {note}\n")
    if manual_map_unmapped:
        wf.write("\nmanual_column.json map rules: source values with no map entry (kept unchanged):\n")
        for (ds_id, target, src_col), vals in manual_map_unmapped.items():
            wf.write(f"  {ds_id} '{src_col}' -> '{target}': {sorted(vals)}\n")
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
for (ds_id, target, src_col), vals in manual_map_unmapped.items():
    print(f"WARNING: {ds_id} map '{src_col}' -> '{target}': no map entry for {sorted(vals)} (kept unchanged)")
print(f"Wrangling notes written to {wrangling_notes_path}")
