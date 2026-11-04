import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--embedding",
    required=True,
    choices=["uce", "uce_brain_512", "uce_brain_768_bican", "uce_brain_768_bican_pt", "scimilarity"],
    help="Embedding type to use (uce, uce_brain_512, uce_brain_768_bican, uce_brain_768_bican_pt or scimilarity)"
)

args = parser.parse_args()
embedding = args.embedding


import pandas as pd
import anndata as ad
try:
    from anndata.io import read_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem
import io
import numpy as np
import os, re, sys, time, json
import boto3
from collections import defaultdict
import h5py
import warnings
from numpy.lib.format import open_memmap

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
    file_suffix = "_uce_adata.h5ad"
    obsm_key = "X_uce"
elif embedding == "uce_brain_512":
    results_dir = "UCE_brain_512_npy_obs"
    duplicate_dir = "duplicate_UCE_brain_512"
    bucket_prefix = "UCE_brain_512/"
    file_suffix = "_ucebrain512_adata.h5ad"
    obsm_key = "X_ucebrain512"
    embedding = "uce"  # output files named *_uce.npy like uce; location, suffix and obsm key differ
elif embedding == "uce_brain_768_bican":
    results_dir = "UCE_brain_768_bican_npy_obs"
    duplicate_dir = "duplicate_UCE_brain_768_bican"
    bucket_prefix = "UCE_brain_768_bican/"
    file_suffix = "_ucebrain768bican_adata.h5ad"
    obsm_key = "X_ucebrain768bican"
    embedding = "uce"  # same as uce_brain_512
elif embedding == "uce_brain_768_bican_pt":
    results_dir = "UCE_brain_768_bican_pt_npy_obs"
    duplicate_dir = "duplicate_UCE_brain_768_bican_pt"
    bucket_prefix = "UCE_brain_768_bican_pt/"
    file_suffix = "_ucebrain768bicanpt_adata.h5ad"
    obsm_key = "X_ucebrain768bicanpt"
    embedding = "uce"  # same as uce_brain_512
elif embedding == "scimilarity":
    results_dir = "SCimilarity_npy_obs"
    duplicate_dir = "duplicate_SCimilarity"
    bucket_prefix = "scimilarity/" 
    file_suffix = "_scimilarity_adata.h5ad"
    obsm_key = "X_scimilarity"
else:
    sys.exit(1)

os.makedirs(results_dir, exist_ok=True)

def saveUCE (adata, prefix_id, results_dir):
    uce_df = pd.DataFrame(adata.obsm[obsm_key])
    np.save(os.path.join(results_dir, f'{prefix_id}_uce.npy'), uce_df)
    adata.obs.to_csv(os.path.join(results_dir, f'{prefix_id}_obs.tsv.gz'), sep='\t', compression='gzip')
    
def saveSCimilarity (adata, prefix_id, results_dir):
    scimilarity_df = pd.DataFrame(adata.obsm[obsm_key])
    np.save(os.path.join(results_dir, f'{prefix_id}_scimilarity.npy'), scimilarity_df)
    adata.obs.to_csv(os.path.join(results_dir, f'{prefix_id}_obs.tsv.gz'), sep='\t', compression='gzip')
    
PER_CHUNK_PROVENANCE = ("pipeline_git_commit", "bf16")

def model_provenance(uns):
    """uns['<method>_provenance'] without the fields that legitimately vary per chunk"""
    return {k: {kk: vv for kk, vv in v.items() if kk not in PER_CHUNK_PROVENANCE}
            for k, v in uns.items() if k.endswith("_provenance")}

def check_chunk_provenance(prefix_id, file, uns, first_uns):
    """Stop on a chunk whose provenance can't be trusted or disagrees with the dataset's first
    chunk. Chunks predating these uns fields (no source_verified / *_provenance) pass."""
    if uns.get("source_verified") is False:
        raise ValueError(f"{prefix_id}: {file} has source_verified=False, so its source_url "
                         f"{uns.get('source_url')} is only what dataset_ids.txt named, not a "
                         f"record of what was staged; re-embed it from a recorded source")
    if first_uns is None:
        return
    a, b = first_uns.get("source_url"), uns.get("source_url")
    if a and b and a != b:
        raise ValueError(f"{prefix_id}: {file} was embedded from {b}, earlier chunks from {a}")
    # same model for every chunk; the pipeline commit may differ between reruns, and bf16
    # with the GPU the job landed on (bf16 and fp32 embeddings are almost identical)
    if model_provenance(uns) != model_provenance(first_uns):
        raise ValueError(f"{prefix_id}: {file} was embedded with {model_provenance(uns)}, "
                         f"earlier chunks with {model_provenance(first_uns)}")

NAN_CHUNK_MAX_FRAC = 0.01

def nan_check(file, x):
    """A chunk with more than NAN_CHUNK_MAX_FRAC NaN embedding rows is a failed embedding run:
    in the uce-brain 768 build, 8 chunks came back 61-90% all-NaN while every other chunk had
    none. Report it so the chunk can be re-embedded. A few NaN rows are single cells the model
    could not embed; they are only printed, and combineEmbedding.py drops them."""
    n_nan = int(np.isnan(x).any(axis=1).sum())
    if n_nan == 0:
        return None
    frac = n_nan / x.shape[0]
    print(f"  NaN: {n_nan:,} / {x.shape[0]:,} rows ({frac:.2%}) in {file}"
          + ("" if frac > NAN_CHUNK_MAX_FRAC else "; under the failed-chunk cutoff, dropped at combine"))
    if frac <= NAN_CHUNK_MAX_FRAC:
        return None
    return {"file": file, "nan_rows": n_nan, "n_cells": int(x.shape[0])}

nan_datasets = {}  # dataset -> chunks with NaN rows, reported at the end of the run

def saveSource(prefix_id, results_dir, unses, nan_chunks=()):
    """Write <id>_provenance.json: the source h5ad URL the embedding was run on (uns['source_url'],
    set by RunFoundationModelNRP/fixed_pipeline.py, or "unknown" if any chunk predates it),
    whether the pipeline verified it, the pipeline
    commits, the model provenance (uns['<method>_provenance']) of the chunks and any chunks
    with NaN embedding rows."""
    sources = [u.get("source_url") for u in unses]
    known = {s for s in sources if s}
    if len(known) > 1:
        raise ValueError(f"{prefix_id}: chunks were embedded from different source h5ads: {sorted(known)}")
    url = known.pop() if known and all(sources) else "unknown"
    if known and url == "unknown":
        warnings.warn(f"{prefix_id}: some chunks have no source_url; writing 'unknown'")

    verified = [u.get("source_verified") for u in unses]
    bf16 = [v.get("bf16") for u in unses for k, v in u.items() if k.endswith("_provenance")]
    provenance = {
        "source_url": url,
        # True only if every chunk's pipeline confirmed the staged copy came from source_url
        "source_verified": all(v is True for v in verified) if None not in verified else None,
        "pipeline_git_commits": sorted({u.get("pipeline_git_commit") or "unknown" for u in unses}),
        "n_chunks": len(unses),
        **({"bf16_chunks": bf16.count(True), "fp32_chunks": bf16.count(False)} if bf16 else {}),
        **model_provenance(unses[0]),
        "nan_chunks": list(nan_chunks),
    }
    if nan_chunks:
        nan_datasets[prefix_id] = list(nan_chunks)
    with open(os.path.join(results_dir, f'{prefix_id}_provenance.json'), 'w') as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")

def result_exists(prefix_id, results_dir, embedding):
    if not os.path.exists(os.path.join(results_dir, f'{prefix_id}_{embedding}.npy')):
        return False
    # a result with NaN chunks is downloaded again, so re-embedded chunks get picked up
    prov_path = os.path.join(results_dir, f'{prefix_id}_provenance.json')
    if os.path.exists(prov_path):
        with open(prov_path) as f:
            if json.load(f).get("nan_chunks"):
                print(f"{prefix_id}: earlier download had NaN chunks; downloading again")
                return False
    return True

class S3File(io.RawIOBase):
    """Read-only, seekable file object over an S3 object using ranged GETs,
    so h5py fetches only the bytes it reads instead of the whole file."""
    def __init__(self, s3, bucket, key):
        self.s3, self.bucket, self.key, self.pos = s3, bucket, key, 0
        self.size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.s3.get_object(
            Bucket=self.bucket, Key=self.key, Range=f"bytes={self.pos}-{self.pos + n - 1}"
        )["Body"].read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

def _plain(v):
    """numpy scalars/bytes from read_elem -> JSON-serializable Python values"""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, bytes):
        return v.decode()
    return v.item() if hasattr(v, "item") else v

def read_from_s3(profile, bucket, key):
    """Read only obs, obsm[obsm_key] and the provenance in uns (source_url, source_verified,
    pipeline_git_commit, <method>_provenance) of an .h5ad on S3 (~40 MB of a ~950 MB file;
    X is never fetched). Returns an in-memory AnnData with just those."""
    s3 = boto3.Session(profile_name=profile).client("s3")
    raw = io.BufferedReader(S3File(s3, bucket, key), buffer_size=4 << 20)
    with h5py.File(raw, "r") as f:
        x = f["obsm"][obsm_key][:]
        obs = read_elem(f["obs"])
        uns = {k: _plain(read_elem(f["uns"][k])) for k in f["uns"].keys()
               if k in ("source_url", "source_verified", "pipeline_git_commit") or k.endswith("_provenance")}
    return ad.AnnData(obs=obs, obsm={obsm_key: x}, uns=uns)

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
    # temporary, one per dataset and process: a fixed name in the current folder let two
    # downloads running in the same (shared) folder overwrite each other's file
    embedding_path_h5 = os.path.join(results_dir, f".download_{dataset_id}_{os.getpid()}.h5")
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
    col_order = None
    start = 0
    unses = []
    nan_chunks = []

    for file in files:
        print("Processing", start, file)

        # --- Read obs and embedding directly from S3 ---
        t0 = time.perf_counter()
        adata = read_from_s3(profile, bucket, f"{bucket_prefix}{file}")
        x = np.array(adata.obsm[obsm_key])  # (n_cells, n_features)
        check_chunk_provenance(dataset_id, file, adata.uns, unses[0] if unses else None)
        unses.append(dict(adata.uns))
        bad = nan_check(file, x)
        if bad:
            nan_chunks.append(bad)

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
                )
            elif embedding == "scimilarity":
                dset = h5f.create_dataset(
                    "scimilarity",
                    shape=(0, n_features),
                    maxshape=(None, n_features),  # allow unlimited rows
                    dtype="float32",
                    chunks=True,
                )

        # --- Resize and append new data ---
        t1 = time.perf_counter()
        dset.resize(dset.shape[0] + n_new, axis=0)
        dset[-n_new:] = x

        # --- Append metadata ---
        t2 = time.perf_counter()
        obs_df = adata.obs.rename(columns={"unique_dataset_id": "dataset_id"})
        if col_order is None:
            col_order = obs_df.columns.tolist()
            obs_df.to_csv(obs_path, sep="\t", mode="w", header=True)
        else:
            missing = [c for c in col_order if c not in obs_df.columns]
            if missing:
                warnings.warn(f"Batch {file} missing columns {missing}; filling with NA")
            obs_df = obs_df.reindex(columns=col_order)
            obs_df.to_csv(obs_path, sep="\t", mode="a", header=False)
        t3 = time.perf_counter()
        print(f"  read {t1 - t0:.1f}s, h5 append {t2 - t1:.1f}s, obs tsv {t3 - t2:.1f}s")

        start += n_new

    # --- Close HDF5 file ---
    h5f.close()
    print(f"HDF5 written to: {embedding_path_h5}")

    # --- Stream HDF5 -> disk-backed .npy in chunks (avoids materializing the
    # full array in RAM, which OOMs for large datasets, e.g. ~46 GiB for
    # ~9.7M cells x 1280 features) ---
    with h5py.File(embedding_path_h5, "r") as f:
        dset_key = "uce" if embedding == "uce" else "scimilarity"
        src = f[dset_key]
        n_rows, n_features = src.shape
        mmap_array = open_memmap(
            embedding_path_npy, mode="w+", dtype="float32", shape=(n_rows, n_features)
        )
        chunk_rows = 200_000
        for row_start in range(0, n_rows, chunk_rows):
            row_end = min(row_start + chunk_rows, n_rows)
            mmap_array[row_start:row_end] = src[row_start:row_end]
        mmap_array.flush()
        del mmap_array
    os.remove(embedding_path_h5)
    saveSource(dataset_id, results_dir, unses, nan_chunks)
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
    
    file_id = file.replace(file_suffix, "")  # such as f0f0d7c4-3bec-428e-9539-c99d36548d96_(cell_0_1000)

    if result_exists(file_id, results_dir, embedding):
        print (f"{file_id} {embedding} results already available in results dir")
        continue
    if result_exists(file_id, duplicate_dir, embedding):
        print (f"{file_id} {embedding} results already available in duplicate dir")
        continue
        
    print (f"{file_id} processing ...")
    # read obs and embedding directly from the s3 bucket
    adata = read_from_s3(profile, bucket, f'{bucket_prefix}{file}')
    print(f"Read {bucket_prefix}{file}")

    # change adata.obs unique_dataset_id to dataset_id
    adata.obs = adata.obs.rename(columns={"unique_dataset_id": "dataset_id"})
    if embedding == "uce":
        saveUCE (adata, file_id, results_dir)
    elif embedding == "scimilarity":
        saveSCimilarity (adata, file_id, results_dir)
    check_chunk_provenance(file_id, file, adata.uns, None)
    bad = nan_check(file, np.asarray(adata.obsm[obsm_key]))
    saveSource(file_id, results_dir, [dict(adata.uns)], [bad] if bad else [])
    print(f"Saved as {results_dir}/{file_id}_{embedding}.npy, _obs.tsv and _provenance.json")

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

if nan_datasets:
    print(f"ERROR: chunks with NaN embedding rows (re-embed them, then rerun; these datasets are "
          f"downloaded again automatically):")
    for ds, chunks in nan_datasets.items():
        for c in chunks:
            print(f"  {ds}: {c['file']}  {c['nan_rows']:,} / {c['n_cells']:,} rows NaN")
    sys.exit(1)
