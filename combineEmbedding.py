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

args = parser.parse_args()
embedding = args.embedding
results_dir = args.results_dir

import cellxgene_census  # python3.11
import os, sys, re
import numpy as np
import pandas as pd
import h5py

obs_only = args.obs_only
uce_only = args.uce_only

if embedding =="uce":
    embedding_outfile = "uce.h5"
    data_dirs =["UCE_npy_obs"]
    n_cols = 1280
elif embedding == "scimilarity":
    embedding_outfile = "scimilarity.h5"
    data_dirs =["SCimilarity_npy_obs"]
    n_cols = 128
else:
    print ("embedding method not supported yet")
    sys.exit(1)
os.makedirs(results_dir, exist_ok=True)

select_datasets = []
if args.select_datasets_file:
    with open(args.select_datasets_file, 'r') as f:
        select_datasets = [line.strip() for line in f if line.strip()]

obs_outfile = "obs.tsv.gz"

ucsc_file = "datasetLists/ucsc_datasets.txt"
cgCensus_non_primary_file ="datasetLists/cgCensus_non_primary_datasets.txt"
ignore_file = "datasetLists/ignore_datasets.txt"

dataset_list = []
obs_list =[]
cellCount = 0
harmonized_datasets = []   # (real_dataset_id, dataset_id, mapped_filepath, doc_filepath) — used cell label harmonization
fallback_datasets = []     # (real_dataset_id, dataset_id) — fell back to cell_type
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
               'harmonized_cell_label_ontology_term_id',
               'harmonized_cell_label',
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

#ucsc info
ucscInfo = get_ucsc_collection()

# file sets
ucsc_datasets = get_ucsc_dataset_ids()
cgCensus_non_primary_datasets = get_cgCensus_non_primary_dataset_ids()
ignore_datasets = get_ignore_dataset_ids()

if not obs_only:
    fuce = h5py.File(os.path.join(results_dir, embedding_outfile), "w")
    dset_uce = fuce.create_dataset("data",
                                   shape=(0, n_cols),
                                   maxshape=(None, n_cols),
                                   dtype=arr_dtype
                                   )

for singleDir in data_dirs:
    for root, dirs, files in os.walk(singleDir):
        for file in files:
            #if counter >1:
            #    break
            if file.endswith('.npy'):
                dataset_id = file.removesuffix("_uce.npy").removesuffix("_scimilarity.npy")
                match = re.match(r'^((?:ucsc-)?[a-f0-9\-]+)_', file)
                real_dataset_id = match.group(1) if match else dataset_id

                if len(select_datasets) !=0 and real_dataset_id not in select_datasets:
                    continue
                if len(select_datasets) ==0 and real_dataset_id in ignore_datasets:
                    continue
                
                print (singleDir, dataset_id)
                if dataset_id not in dataset_list:
                    dataset_list.append(dataset_id)

                    uce_filepath = os.path.join(singleDir, file)
                    uce  = np.load(uce_filepath)
                    obs_filepath = os.path.join(singleDir, dataset_id+"_obs.tsv")
                    obs = pd.read_csv(obs_filepath, sep='\t', low_memory=False)

                    # Auto-detect harmonization: if a remap file exists, load the
                    # mapped obs and add harmonized_cell_type* columns.
                    remap_filepath = os.path.join(singleDir, dataset_id+"_remap.tsv")
                    mapped_filepath = os.path.join(singleDir, dataset_id+"_obs_mapped.tsv.gz")
                    doc_filepath = os.path.join(singleDir, dataset_id+"_obs_mapped_doc.txt")
                    if os.path.exists(remap_filepath) and os.path.exists(mapped_filepath) and os.path.exists(doc_filepath):
                        print(f"  Harmonizing {dataset_id} from {mapped_filepath}")
                        mapped_obs = pd.read_csv(mapped_filepath, sep='\t', low_memory=False)
                        obs["harmonized_cell_label_ontology_term_id"] = mapped_obs["mapped_cell_label_ontology_term_id"].values
                        obs["harmonized_cell_label"] = mapped_obs["mapped_cell_label"].values
                        harmonized_datasets.append((real_dataset_id, dataset_id, mapped_filepath, doc_filepath))
                    else:
                        if "harmonized_cell_label_ontology_term_id" in obs.columns:
                            obs["harmonized_cell_label_ontology_term_id"] = obs["harmonized_cell_label_ontology_term_id"]
                            obs["harmonized_cell_label"] = obs["harmonized_cell_label"]
                        else:
                            obs["harmonized_cell_label_ontology_term_id"] = obs["cell_type_ontology_term_id"]
                            obs["harmonized_cell_label"] = obs["cell_type"]
                        fallback_datasets.append((real_dataset_id, dataset_id))

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
                                obs["collection_doi_label"] = cxgInfo[cxgInfo.dataset_id == dataset_id]["collection_doi_label"].values[0]
                                obs["collection_doi"] = cxgInfo[cxgInfo.dataset_id == dataset_id]["collection_doi"].values[0]
                                obs["dataset_title"] = cxgInfo[cxgInfo.dataset_id == dataset_id]["dataset_title"].values[0]
                            else:
                                obs["collection_doi"] = ucscInfo.at[dataset_id, "collection_doi"]
                                obs["collection_doi_label"] = ucscInfo.at[dataset_id, "collection_doi_label"]
                                obs["dataset_title"] = ucscInfo.at[dataset_id, "dataset_title"]
                            obs_list.append(obs)
                                                    
                        ## export to dest_uce, one numpy array at a time concatenate at the end, memory efficient and fast
                        if not obs_only:
                            dset_uce.resize(dset_uce.shape[0] + uce.shape[0], axis=0)
                            dset_uce[-uce.shape[0]:] = uce
                        
                    else:
                        print ("All embedding data are NaN (very rare) or is_primary_data are all False (more likely)")
                else:
                    print ("Duplicate dataset")
if not obs_only:
    fuce.close()

if not uce_only:
    print("Start with obs processing")
    combined_obs = pd.concat(obs_list, axis=0, join='outer')

    ## shape
    num_cols = combined_obs.shape[1]

    print ("combined obs shape:", (cellCount, num_cols))

    #export combined_obs
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
    wf.write("Add column 'harmonized_cell_type', initialized from 'cell_type'; "
             "overwritten from 'mapped_cell_label' for datasets with cell label harmonization output "
             "(*_remap.tsv + *_obs_mapped.tsv.gz)\n")
    wf.write("Add column 'harmonized_cell_type_ontology_term_id', initialized from 'cell_type_ontology_term_id'; "
             "overwritten from 'mapped_cell_label_ontology_term_id' for datasets with cell label harmonization output "
             "(*_remap.tsv + *_obs_mapped.tsv.gz)\n")
    wf.write(f"\nDatasets processed: {len(dataset_list)}\n")
    if harmonized_datasets:
        wf.write(f"\nCell label harmonization applied ({len(harmonized_datasets)} datasets):\n")
        for real_ds_id, ds_id, mapped_fp, doc_fp in harmonized_datasets:
            doi, title = _dataset_info(ds_id, ucsc_datasets, cxgInfo, ucscInfo)
            wf.write(f"\n  {real_ds_id}  '{title}'  {doi}\n")
            if os.path.exists(doc_fp):
                with open(doc_fp) as doc_f:
                    for line in doc_f:
                        wf.write("    " + line)
                wf.write("\n")
    if fallback_datasets:
        wf.write(f"\nCell label harmonization not available — fallback to cell_type ({len(fallback_datasets)} datasets):\n")
        for real_ds_id, ds_id in fallback_datasets:
            doi, title = _dataset_info(ds_id, ucsc_datasets, cxgInfo, ucscInfo)
            wf.write(f"  {real_ds_id}  '{title}'  {doi}\n")
print(f"Wrangling notes written to {wrangling_notes_path}")
