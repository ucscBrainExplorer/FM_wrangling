# Build a UCE reference atlas

Given a set of wrangled h5ads with UCE embeddings, build a PUMAP reference atlas for the browser.

## Overview

The pipeline has two phases:

**Phase 1 — Build the reference** (this skill)
1. Combine embeddings with `combineEmbedding.py --shuffle`, then annotate obs with `annotate_obs.py`
2. PCA + KNN (run in parallel)
3. Train PUMAP
4. Build pyramid
5. Add the reference to data-explorer

**Phase 2 — Map query data onto the reference**
- Use `map_newdata.py` to project new datasets onto the finished reference → arrow files

---

## Step 1 — Combine embeddings (or shuffle a single dataset)

### Multiple datasets — use `combineEmbedding.py`

Run from `/mnt/efsUCE/`:

```bash
python3.11 FM_wrangling/combineEmbedding.py \
    --embedding uce \
    --results_dir <ATLAS_DIR>/ \
    --select-datasets-file <ATLAS_DIR>/dataset_ids.tsv \
    --shuffle
```

- `dataset_ids.tsv` — one dataset per line, two tab-separated columns, no header: dataset ID (matching npy filenames in
  `UCE_npy_obs/`) and the exact source h5ad URL the embedding was run on (`s3://…` or `https://…`), e.g.
  `8e10f1c4-8e98-41e5-b65f-8cd89a887122<TAB>s3://cellxgene-census-public-us-west-2/cell-census/2025-11-17/h5ads/8e10f1c4-8e98-41e5-b65f-8cd89a887122.h5ad`.
  The URL is recorded automatically: `RunFoundationModelNRP/fixed_pipeline.py` stores it in each output h5ad as
  `uns["source_url"]`, `getEmbeddingfromS3.py` writes it as `source_url` in `<id>_provenance.json` next to the npy, and `combineEmbedding.py`
  takes it from there when column 2 is empty, stops if a URL is missing or the two disagree, and writes the filled-in
  file to `<ATLAS_DIR>/dataset_ids.tsv`. Datasets embedded before this have `source_url` = `unknown` (or no `_provenance.json`), so
  give their URL in column 2, with its release date, never "latest". `annotate_obs.py` takes the URLs for
  `metadata_summary.json` from this file and stops if any dataset is missing one.
- `--shuffle` writes a new `shuffle_index.npy` and shuffles `uce.h5` and `obs.tsv.gz` in sync
- Verify output: cell count matches expectation, `obs.tsv.gz` contains expected columns

On startup the script validates that:
- All datasets in `duplicate_UCE/` (or `duplicate_SCimilarity/`) are listed in `datasetLists/ignore_datasets.txt`
- All `ucsc-*` prefixed datasets in `UCE_npy_obs/` and `duplicate_UCE/` are listed in `datasetLists/ucsc_datasets.txt`

### Single dataset — use `shuffle.py` directly

If only one dataset needs to be shuffled (no combine step needed):

```bash
/usr/bin/python3.11 FM_wrangling/utils/shuffle.py \
    --uce-in UCE_npy_obs/<dataset_id>_uce.npy \
    --obs-in UCE_npy_obs/<dataset_id>_obs.tsv \
    --out-dir <ATLAS_DIR>
```

Writes `<ATLAS_DIR>/uce.h5` (uncompressed), `obs.tsv.gz` and `shuffle_index.npy`. Obs is staged as Parquet in `--nvme-dir` (default `/nvme`) and values are kept verbatim.

Use `--chunked` if RAM is constrained: RAM drops to ~32 bytes/row plus one read block, but `--nvme-dir` needs free space equal to the full uncompressed array. `--chunked` requires `.h5` input; if the source h5 has tiny chunks (e.g. `combined_UCE_56M/uce.h5`), copy it to `/nvme` first and shuffle the copy.

### Annotate obs and write the metadata summary — `annotate_obs.py`

Run from the same directory as `combineEmbedding.py`, after it:

```bash
python3 FM_wrangling/annotate_obs.py --data_dir <ATLAS_DIR>
```

- Reads `<ATLAS_DIR>/obs.tsv.gz` and `dataset_ids.tsv`, plus `datasetLists/` (`ucsc_datasets.txt`, `manual_column.json`, newest `datasets.info.cxg.*.txt`).
- Writes `obs_annotated.tsv.gz`, `metadata_summary.json`, `wrangling_notes_annotation.txt` and the two `system_prompt_*.txt` files.
- Normalizes organism and development stage (age bucket) with OpenAI gpt-4.1; needs `OPENAI_API_KEY` in `.env` in the run directory. The two mappings are printed for checking.
- Keeps the hand-filled fields of an existing `metadata_summary.json` (`reference_name`, `reference_uuid`, `version_label`, `reference_pyramid`, `foundation_model`, `abstract`); everything else is regenerated.
- `--output_dir <DIR>` writes the outputs elsewhere (still reading the hand-filled fields from `<ATLAS_DIR>`), to compare a re-run before replacing the files.

## Step 2 — PCA and KNN (run in parallel)

### KNN (NRP cluster)

First copy `uce.h5` to S3:
```bash
aws s3 cp scpca_glioma_shuffled/uce.h5 s3://latentbrain/scpca_glioma_shuffled/uce.h5
```

Then submit two KNN jobs (ivfflat and ivfpq_320) to the NRP cluster:
```bash
cd ../RunKnnNRP
bash run_pipeline_knn_job.sh
```

The script is pre-configured for `DIR_NAME=scpca_glioma_shuffled` and `EMBEDDING=uce`. It submits Kubernetes jobs that pull `uce.h5` from `s3://latentbrain/scpca_glioma_shuffled/` and build the KNN indices.

When the jobs complete, copy outputs back:
```bash
aws s3 cp s3://latentbrain/scpca_glioma_shuffled/knn_30.npz scpca_glioma_shuffled/
aws s3 cp s3://latentbrain/scpca_glioma_shuffled/index_ivfflat.faiss scpca_glioma_shuffled/
aws s3 cp s3://latentbrain/scpca_glioma_shuffled/index_ivfpq_320.faiss scpca_glioma_shuffled/
```

Outputs: `knn_30.npz`, `index_ivfflat.faiss`, `index_ivfpq_320.faiss`

### PCA

```bash
python3 FM_wrangling/runIPCA.py --results_dir <ATLAS_DIR>
```

Runs IncrementalPCA (50 components) on `uce.h5` in streaming chunks. Options:
- `--n_components` — number of PCA components (default: 50)
- `--chunk_size` — rows per chunk (default: 100,000; set to 10,000 for small atlases <100k cells to avoid an h5py chunk shape error)
- `--group_name` — optional suffix for output filenames

Outputs: `ipca_model.pkl`, `pca.h5`

On a shared machine (e.g. courtyard), cap the BLAS threads — by default NumPy uses every core (a 7.1M × 512 run took ~34 of courtyard's 64 cores), and IncrementalPCA gains little beyond ~16:
```bash
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    python3 FM_wrangling/runIPCA.py --results_dir <ATLAS_DIR>
```

## Step 3 — Train PUMAP

Requires a GPU machine (spin up an AWS GPU instance manually). Run via Jupyter notebook.

Open `/mnt/efsUCE/3_trainPUMAPmodel.ipynb`. Set parameters at the top:
```python
results_dir = "/mnt/efsUCE/scpca_glioma_shuffled"
existing_model_iteration = 0   # 0 = start fresh; N = resume from pytorch_eN.pkl
epochs = 10                    # epochs to train (or add) in this run
k = 10                         # KNN neighbors used for PUMAP (subset of knn_30)
```

The notebook trains PUMAP on the PCA-transformed embeddings using the precomputed KNN. Training is incremental — run multiple times to add epochs, saving a checkpoint each time (`pytorch_e{total}.pkl`). After training it transforms all cells and auto-detects if axes need swapping (width > height).

Outputs per run:
- `pytorch_e{N}.pkl` — PUMAP model checkpoint
- `transform_e{N}.json` — transform config used by `map_newdata.py` (contains paths to PUMAP model, ipca model, swap_xy_matrix)
- `umap_e{N}.npy` — 2D UMAP coordinates for all reference cells

## Step 4 — Build pyramid

### Generate pyramid config

Run from `/mnt/efsUCE/`:
```bash
python3 FM_wrangling/buildPyramidConfig.py \
    --results_dir <ATLAS_DIR> \
    --umap_file umap_e{N}.npy
```

This auto-generates `<ATLAS_DIR>/pyramid_e{N}_config.json` with all eligible obs columns ranked by biological relevance. HIGH_PRIORITY columns (`harmonized_cell_label`, `sample_label`, `author_label`, `disease`, `suspension_type`, `assay`, `development_stage`, `tissue`, `collection_doi_label`) are always included regardless of cardinality or null rate. `harmonized_cell_label` is pinned first; `sample_label` is pinned second.

Fill in before running the notebook:
- `reference_name` — human-readable name
- `target_dir` — where tiles will be written
- These can be copied from a prior config if rebuilding

### Clear old target directory

Clear everything in `target_dir` — PNGs, arrow files, and metadata.json all belong to the old reference and are no longer valid:
```bash
find <target_dir> -mindepth 1 -delete
```

### Run the notebook

Open `/mnt/efsUCE/5_buildPyramid.ipynb`. Set at the top:
```python
results_dir = "<ATLAS_DIR>"
pyramid_file = "pyramid_e{N}_config.json"
```

The notebook auto-selects tile resolution to keep pixel loss rate <3%, builds multi-level PNG tiles for each feature, and writes `metadata.json` to `target_dir`.

Output: ~60 PNG tiles per feature + `metadata.json` in `target_dir`

## Step 5 — Add the reference to data-explorer — `merge_metadata_summary.py`

Set `reference_pyramid` in `<ATLAS_DIR>/metadata_summary.json` to the pyramid's public URL, then merge it into the data-explorer metadata list:

```bash
python3 FM_wrangling/merge_metadata_summary.py data-explorer/UCSC_Brain_Explorer_metadata.json <ATLAS_DIR>/metadata_summary.json
```

- Matches entries by `reference_uuid`: an existing uuid is replaced whole (each field change is printed), a new uuid is appended.
- Writes a timestamped `.bak` of the target next to it; delete it once the diff looks right.

Copy the notes the site shows into `data-explorer/wrangling_notes/<reference_uuid>/`:
- `<ATLAS_DIR>/wrangling_notes_annotation.txt` → `wrangling_notes.txt`
- `system_prompt_normalized_age_bucket.txt` and `system_prompt_normalized_organism.txt`

---

## Running on courtyard (GI shared machine)

- **Working directory:** run everything from `/public/groups/hausslerlab/people/jzhu2` (repo, `datasetLists/`, `UCE_brain_512_npy_obs/` live there). The home directory has a ~31 GB quota; large outputs there fail with "Disk quota exceeded".
- **Staging:** there is no `/nvme`; pass `--staging-dir /data/scratch/jzhu2` to `combineEmbedding.py` (local disk, files not accessed for 14 days are deleted automatically).
- **Threads:** cap NumPy/sklearn jobs with `OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16` (see PCA above).
- **No AVX:** the CPU (Xeon E7-4830) cannot run `tiledbsoma` / `cellxgene_census` ("Illegal instruction"). `combineEmbedding.py` reads CELLxGENE metadata from the newest `datasetLists/datasets.info.cxg.*.txt` instead; refresh it with `datasetLists/fetch_cxg_datasets.py`.
- **Long jobs:** start them in `screen` so they survive the terminal closing.

### UCE brain 512 (`--embedding uce_brain_512`)

- Inputs come from `UCE_brain_512_npy_obs/`; embeddings are 512-dim instead of 1280.
- Example: `combined_UCE_5neuro_ucebrain512_shuffle_7253ea1e-e5ff-4f93-80fa-c108806fc7c6/` (5 brain datasets, 7.1M cells, ~16 min with `--shuffle`):
  ```bash
  python3 FM_wrangling/combineEmbedding.py --embedding uce_brain_512 \
      --results_dir combined_UCE_5neuro_ucebrain512_shuffle_7253ea1e-e5ff-4f93-80fa-c108806fc7c6 \
      --select-datasets-file combined_UCE_5neuro_ucebrain512_shuffle_7253ea1e-e5ff-4f93-80fa-c108806fc7c6/dataset_ids.tsv \
      --obs-columns-file combined_UCE_5neuro_ucebrain512_shuffle_7253ea1e-e5ff-4f93-80fa-c108806fc7c6/obs_columns_brain.txt \
      --staging-dir /data/scratch/jzhu2 --shuffle
  ```
- `--obs-columns-file` (select mode only) limits obs to the listed columns, in file order; without it, select mode writes the union of all columns.
- `datasetLists/manual_column.json` applies in both modes; rule types `static`, `column` and `map` (value translation, `null` = empty), each with an optional `note` that is copied into `wrangling_notes_combineEmbedding.txt`.
- KNN: use `--index_type ivfflat` or `ivfpq_128`; `ivfpq_320` fails because 512 is not divisible by 320.
- PCA/PUMAP models trained on 1280-dim UCE do not apply; retrain on the 512-dim `uce.h5`, and map only query data embedded with the same UCE brain model.

### UCE brain 768 BICAN (`--embedding uce_brain_768_bican`)

- Same flow as UCE brain 512, for the `ucebrain768bican` model (UCE-brain d768l14v2 + BICAN anneal) in RunFoundationModelNRP.
- `getEmbeddingfromS3.py --embedding uce_brain_768_bican` reads `s3://latentbrain/UCE_brain_768_bican/*_ucebrain768bican_adata.h5ad` (obsm `X_ucebrain768bican`) into `UCE_brain_768_bican_npy_obs/`; `combineEmbedding.py` writes a 768-dim `uce.h5`.
- KNN: 768 is divisible by 128 and 256 but not 320, so `ivfpq_320` fails here too.

### Provenance (all embeddings)

- `getEmbeddingfromS3.py` writes `<id>_provenance.json` (source h5ad URL, `source_verified`, pipeline commits, bf16/fp32 chunk counts, and the model's `uns["<method>_provenance"]`, e.g. uce-brain commit) next to each `.npy`.
- It stops on a chunk with `source_verified = False` (embedded from a staged copy with no record of where it came from), or whose source URL or model provenance differs from the dataset's other chunks. bf16 may differ (it follows the GPU; bf16 and fp32 embeddings are almost identical). Chunks from before these fields existed pass.
- It also checks every chunk for NaN embedding rows. A chunk with more than 1% NaN rows is a failed embedding run (the uce-brain 768 build had 8 chunks 61-90% NaN): it is listed under `nan_chunks` in `_provenance.json`, and the run exits 1 at the end. Datasets with `nan_chunks` are downloaded again on the next run, so re-embed the chunks upstream and rerun. `combineEmbedding.py` (select mode) stops before combining if any selected dataset has `nan_chunks`.
- Chunks with 1% or fewer NaN rows are only printed. `combineEmbedding.py` drops those cells, logs how many per dataset, and checks that cells written + NaN-dropped equals the pre-scan count.
- In select mode `combineEmbedding.py` collects them into `<results_dir>/dataset_provenance.json` (null for datasets with no `_provenance.json`) and warns if datasets were embedded with different model provenance.

## Reference

- Script: `/mnt/efsUCE/FM_wrangling/combineEmbedding.py`
- Query mapping: `/mnt/efsUCE/FM_wrangling/map_newdata.py`
- Example atlas: `/mnt/efsUCE/scpca_glioma_shuffled/`
