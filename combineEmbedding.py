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
    help="Path to fast local NVMe scratch space used for temporary memmap during shuffle (default: /nvme)"
)

args = parser.parse_args()

if args.shuffle and (args.obs_only or args.uce_only):
    print("ERROR: --shuffle cannot be used with --obs-only or --uce-only — obs and uce must be written together to stay in sync.")
    sys.exit(1)

shuffle = args.shuffle
embedding = args.embedding
results_dir = args.results_dir
nvme_dir = args.nvme_dir

import cellxgene_census  # python3.11
import os, sys, re
import numpy as np
import pandas as pd
import h5py

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
obs_list =[]
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
        fuce = h5py.File(os.path.join(results_dir, embedding_outfile), "w")
        dset_uce = fuce.create_dataset("data",
                                       shape=(0, n_cols),
                                       maxshape=(None, n_cols),
                                       dtype=arr_dtype
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
            uce  = np.load(uce_filepath)
            obs_filepath = os.path.join(singleDir, dataset_id+"_obs.tsv")
            if not os.path.exists(obs_filepath):
                obs_filepath = os.path.join(singleDir, dataset_id+"_obs.tsv.gz")
            obs = pd.read_csv(obs_filepath, sep='\t', low_memory=False)

            # If a UCSC remap file exists, merge mapped obs columns in as-is
            remap_filepath = os.path.join(singleDir, dataset_id+"_remap.tsv")
            mapped_filepath = os.path.join(singleDir, dataset_id+"_obs_mapped.tsv.gz")
            doc_filepath = os.path.join(singleDir, dataset_id+"_obs_mapped_doc.txt")
            if not os.path.exists(doc_filepath):
                doc_filepath = os.path.join(singleDir, dataset_id+".report.md")
            if os.path.exists(remap_filepath) and os.path.exists(mapped_filepath) and os.path.exists(doc_filepath):
                print(f"  Merging UCSC remap for {dataset_id} from {mapped_filepath}")
                mapped_obs = pd.read_csv(mapped_filepath, sep='\t', low_memory=False)
                for col in mapped_obs.columns:
                    if col not in obs.columns:
                        obs[col] = mapped_obs[col].values
                remapfile_exist_datasets.append((real_dataset_id, dataset_id, mapped_filepath, doc_filepath))

            # First condition: no NaNs in the row
            mask_uce = ~np.isnan(uce).any(axis=1) # Keep rows where no NaN is present
            # Second condition: 'is_primary_data' column is True
            if "is_primary_data" in obs.columns and \
               dataset_id not in ucsc_datasets and \
               dataset_id not in cgCensus_non_primary_datasets :
                mask_primary = (obs["is_primary_data"] == True).values
            else:
                mask_primary = np.ones(len(obs), dtype=bool)  # Keep all rows if column doesn't exist
            
            # Combine the two condition masks
            combined_mask = mask_uce & mask_primary
            if combined_mask.sum() > 1:
                counter = counter + 1
                uce = uce[combined_mask]
                obs = obs[combined_mask]

                cellCount = cellCount + uce.shape[0]
                print (counter, "total cell count:", cellCount)

                # handle obs: index
                if not uce_only:
                    obs['new_index'] = real_dataset_id + "_" + obs.index.astype(str)
                    obs.rename(columns={"Unnamed: 0": "old_index"}, inplace=True)
                    obs.set_index('new_index', inplace=True)
                    obs = obs.assign(dataset_id = real_dataset_id)
                    if len(select_datasets) ==0: # for combining all datasets, have to select columns
                        cols_to_use = [c for c in select_cols if c in obs.columns]
                        obs = obs[cols_to_use]
                    # add collection info
                    if dataset_id not in ucsc_datasets:
                        cxg_row = cxgInfo[cxgInfo.dataset_id == dataset_id]
                        if len(cxg_row) > 0:
                            obs["collection_doi_label"] = cxg_row["collection_doi_label"].values[0]
                            obs["collection_doi"] = cxg_row["collection_doi"].values[0]
                            obs["dataset_title"] = cxg_row["dataset_title"].values[0]
                        else:
                            disc = get_discovery_info(dataset_id)
                            if disc is not None:
                                obs["collection_doi_label"] = disc["collection_doi_label"]
                                obs["collection_doi"] = disc["collection_doi"]
                                obs["dataset_title"] = disc["dataset_title"]
                            else:
                                print(f"  WARNING: {dataset_id} not found in Census or Discovery; collection fields set to NaN")
                                obs["collection_doi_label"] = np.nan
                                obs["collection_doi"] = np.nan
                                obs["dataset_title"] = np.nan
                    else:
                        obs["collection_doi"] = ucscInfo.at[dataset_id, "collection_doi"]
                        obs["collection_doi_label"] = ucscInfo.at[dataset_id, "collection_doi_label"]
                        obs["dataset_title"] = ucscInfo.at[dataset_id, "dataset_title"]
                    obs_list.append(obs)
                                            
                if not obs_only:
                    if shuffle:
                        uce_mmap[uce_write_pos:uce_write_pos + uce.shape[0]] = uce
                        uce_write_pos += uce.shape[0]
                    else:
                        dset_uce.resize(dset_uce.shape[0] + uce.shape[0], axis=0)
                        dset_uce[-uce.shape[0]:] = uce
                
            else:
                print ("All embedding data are NaN (very rare) or is_primary_data are all False (more likely)")
        else:
            print ("Duplicate dataset")
if not obs_only and not shuffle:
    fuce.close()

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
    print("Start with obs processing")
    combined_obs = pd.concat(obs_list, axis=0, join='outer')

    num_cols = combined_obs.shape[1]
    print("combined obs shape:", (cellCount, num_cols))

    if shuffle:
        print("Shuffling obs...")
        combined_obs = combined_obs.iloc[shuffle_idx].reset_index(drop=True)

    combined_obs.to_csv(os.path.join(results_dir, obs_outfile), sep='\t')

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
