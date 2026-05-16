import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--embedding",
    required=True,
    choices=["uce", "scimilarity"],
    help="Embedding type to use (uce or scimilarity)"
)

args = parser.parse_args()
embedding = args.embedding


import pandas as pd
import scanpy as sc
import numpy as np
import os, re, sys
import boto3
from collections import defaultdict
import h5py
import warnings

#NRP bucket
#bucket = "braingeneersdev"
#bucket_prefix = "uce-output-files/"
#profile = "braingeneers"

#S3 bucket
bucket = "latentbrain"
profile = "default"

if embedding =="uce":
    results_dir = "UCE_npy_obs"
    duplicate_dir = "duplicate_UCE"
    bucket_prefix = "UCE/"
elif embedding == "scimilarity":
    results_dir = "SCimilarity_npy_obs"
    duplicate_dir = "duplicate_SCimilarity"
    bucket_prefix = "scimilarity/" 
else:
    sys.exit(1)
    
def saveUCE (adata, prefix_id, results_dir):
    uce_df = pd.DataFrame(adata.obsm['X_uce'])
    np.save(os.path.join(results_dir, f'{prefix_id}_uce.npy'), uce_df)
    adata.obs.to_csv(os.path.join(results_dir, f'{prefix_id}_obs.tsv.gz'), sep='\t', compression='gzip')
    
def saveSCimilarity (adata, prefix_id, results_dir):
    scimilarity_df = pd.DataFrame(adata.obsm['X_scimilarity'])
    np.save(os.path.join(results_dir, f'{prefix_id}_scimilarity.npy'), scimilarity_df)
    adata.obs.to_csv(os.path.join(results_dir, f'{prefix_id}_obs.tsv.gz'), sep='\t', compression='gzip')
    
def result_exists(prefix_id, results_dir, embedding):
    if os.path.exists(os.path.join(results_dir, f'{prefix_id}_{embedding}.npy')):
        return True
    else:
        return False

def get_NRP_h5ad_filenames():
    result=[]
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=bucket_prefix):
        for obj in page.get("Contents", []):
            filename = obj["Key"].removeprefix(bucket_prefix)
            result.append(filename)
    return result

def group_files_by_prefix(filenames):
    """
    Group filenames using patterns:
      1. '<prefix>_cell_<num>_<num>_'
      2. '<prefix>_batch<num>_'
    Non-matching files are collected separately.
    """
    pattern_cell = re.compile(r'^(?P<prefix>.+?)_cell_(\d+)_(\d+)_')
    pattern_batch = re.compile(r'^(?P<prefix>.+?)_batch(\d+)_')

    grouped = defaultdict(lambda: {"files": [], "count": 0})
    non_matching = []

    for f in filenames:
        if not f.strip():  # skip empty strings or whitespace
            continue

        match_cell = pattern_cell.match(f)
        match_batch = pattern_batch.match(f)

        if match_cell:
            prefix = match_cell.group('prefix')
            end = int(match_cell.group(3))
            grouped[prefix]["files"].append(f)
            grouped[prefix]["count"] = max(grouped[prefix]["count"], end)

        elif match_batch:
            prefix = match_batch.group('prefix')
            batch_num = int(match_batch.group(2))
            grouped[prefix]["files"].append(f)
            grouped[prefix]["count"] = max(grouped[prefix]["count"], batch_num)

        else:
            non_matching.append(f)

    # --- Check for continuous ranges ---
    bad_prefixes = []
    for prefix, data in grouped.items():
        files = data["files"]
        ranges = []

        for f in files:
            m_cell = pattern_cell.match(f)
            m_batch = pattern_batch.match(f)

            if m_cell:
                start, end = int(m_cell.group(2)), int(m_cell.group(3))
                ranges.append((start, end, f))
            elif m_batch:
                num = int(m_batch.group(2))
                # Treat batch files as single-point ranges (num,num)
                ranges.append((num, num, f))
            else:
                # Non-matching files should not be here, but just in case
                continue

        # Sort ranges by start
        ranges.sort(key=lambda x: x[0])

        # Replace files list with sorted filenames
        # so that cell order is the same as original
        data["files"] = [f for _, _, f in ranges]

        # Check continuity
        for i in range(1, len(ranges)):
            if ranges[i][0] != ranges[i-1][1]+1:
                warnings.warn(
                    f"Discontinuous cell range in prefix '{prefix}': "
                    f"{ranges[i-1]} followed by {ranges[i]}"
                )
                if prefix not in bad_prefixes:
                    bad_prefixes.append(prefix)
                
    # --- Remove the problematic entries ---
    for prefix in bad_prefixes:
        del grouped[prefix]
        
    return grouped, non_matching

def process_files_via_h5 (
    files,
    profile,
    bucket,
    bucket_prefix,
    results_dir,
    dataset_id,
):
    # --- Setup paths ---
    embedding_path_h5 = f"download_{embedding}.h5"
    if embedding == "uce":
        embedding_path_npy = os.path.join(results_dir, f"{dataset_id}_uce.npy")
    elif embedding == "scimilarity":
        embedding_path_npy = os.path.join(results_dir, f"{dataset_id}_scimilarity.npy")
    obs_path = os.path.join(results_dir, f"{dataset_id}_obs.tsv")

    if os.path.exists(obs_path):
        os.remove(obs_path)
        print(f"Removed existing {obs_path}")

    # --- Create HDF5 dataset (dynamically resizable) ---
    h5f = h5py.File(embedding_path_h5, "w")
    dset = None
    start = 0

    for file in files:
        print("Processing", start, file)

        # --- Download file from S3 ---
        local_path = f"download_{embedding}.h5ad"
        session = boto3.Session(profile_name=profile)
        s3 = session.client("s3")
        s3.download_file(Bucket=bucket, Key=f"{bucket_prefix}{file}", Filename=local_path)

        # --- Read from .h5ad ---
        adata = sc.read_h5ad(local_path, backed="r")
        if embedding == "uce":
            x = adata.obsm["X_uce"]  # (n_cells, n_features)
        elif embedding == "scimilarity":
            x = adata.obsm["X_scimilarity"]  # (n_cells, n_features)

        n_new, n_features = x.shape

        # --- Initialize dataset if first chunk ---
        if dset is None:
            if embedding == "uce":
                dset = h5f.create_dataset(
                    "uce",
                    shape=(0, n_features),
                    maxshape=(None, n_features),  # allow unlimited rows
                    dtype="float32",
                    chunks=True,
                    compression="gzip",
                )
            elif embedding == "scimilarity":
                dset = h5f.create_dataset(
                    "scimilarity",
                    shape=(0, n_features),
                    maxshape=(None, n_features),  # allow unlimited rows
                    dtype="float32",
                    chunks=True,
                    compression="gzip",
                )

        # --- Resize and append new data ---
        dset.resize(dset.shape[0] + n_new, axis=0)
        dset[-n_new:] = x

        # --- Append metadata ---
        mode = "a" if os.path.exists(obs_path) else "w"
        header = not os.path.exists(obs_path)
        obs_df = adata.obs.rename(columns={"unique_dataset_id": "dataset_id"})
        obs_df.to_csv(obs_path, sep="\t", mode=mode, header=header)

        start += n_new

    # --- Close HDF5 file ---
    h5f.close()
    print(f"HDF5 written to: {embedding_path_h5}")

    # --- Load back into NumPy and save as .npy ---
    with h5py.File(embedding_path_h5, "r") as f:
        if embedding == "uce":
            embedding_array = f["uce"][:]
        elif embedding == "scimilarity":
            embedding_array = f["scimilarity"][:]
            
    np.save(embedding_path_npy, embedding_array)
    print(f"Saved final NumPy array: {embedding_path_npy}")
    print(f"Saved metadata TSV: {obs_path}")

    return embedding_path_npy, obs_path


embedding_files = get_NRP_h5ad_filenames()
grouped, others = group_files_by_prefix(embedding_files)

# for just one file per dataset
for file in others:
    match = re.match(r'^(?:ucsc-)?([a-fA-F0-9\-]+)_', file)
    dataset_id = match.group(1) if match else re.sub(r'\.h5ad$', '', file)
    print (dataset_id)
    
    if embedding == "uce":
        file_id = file.replace("_uce_adata.h5ad", "")  # such as f0f0d7c4-3bec-428e-9539-c99d36548d96_(cell_0_1000)
    elif embedding == "scimilarity":
        file_id = file.replace("_scimilarity_adata.h5ad", "")  # such as f0f0d7c4-3bec-428e-9539-c99d36548d96_(cell_0_1000)

    if result_exists(file_id, results_dir, embedding):
        print (f"{file_id} {embedding} results already available in results dir")
        continue
    if result_exists(file_id, duplicate_dir, embedding):
        print (f"{file_id} {embedding} results already available in duplicate dir")
        continue
        
    print (f"{file_id} processing ...")
    #download the file from s3 bucket
    local_path = f"download_{embedding}.h5ad"   # where to save locally
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")
    s3.download_file(Bucket=bucket, Key=f'{bucket_prefix}{file}', Filename=local_path)
    print(f"Downloaded {bucket_prefix}/{file} → {local_path}")

    adata = sc.read_h5ad(local_path, backed="r")
    # change adata.obs unique_dataset_id to dataset_id
    adata.obs = adata.obs.rename(columns={"unique_dataset_id": "dataset_id"})
    if embedding == "uce":
        saveUCE (adata, file_id, results_dir)
    elif embedding == "scimilarity":
        saveSCimilarity (adata, file_id, results_dir)
    print(f"Saved as {results_dir}/{file_id}_{embedding}.npy and _obs.tsv")

# for grouped files
for dataset_id, data in grouped.items():
    files = data["files"]
    cell_count = data["count"]

    print(dataset_id)

    if result_exists(dataset_id, results_dir, embedding):
        print (f"{dataset_id} {embedding} results already available in results dir")
        continue
    if result_exists(dataset_id, duplicate_dir, embedding):
        print (f"{dataset_id} {embedding} results already available in duplicate dir")
        continue

    embedding_path_npy, obs_path = process_files_via_h5( 
        files,
        profile,
        bucket,
        bucket_prefix,
        results_dir,
        dataset_id,
    )
    
    print(f"Saved as {results_dir}/{dataset_id}_{embedding}.npy and _obs.tsv")

