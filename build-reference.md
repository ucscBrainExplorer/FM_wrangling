# Build a UCE reference atlas

Given a set of wrangled h5ads with UCE embeddings, build a PUMAP reference atlas for the browser.

## Overview

The pipeline has two phases:

**Phase 1 — Build the reference** (this skill)
1. Get and combine embeddings: `getEmbeddingfromS3.py`, `combineEmbedding.py --shuffle`
2. PCA + KNN (run in parallel)
3. Train PUMAP
4. Review obs, update `manual_column.json`, annotate with `annotate_obs.py`
5. Build pyramid
6. Add the reference to data-explorer

On courtyard, `run_build.sh` runs Step 1 and the PCA of Step 2 in one go
([below](#steps-12-in-one-go-on-courtyard--run_buildsh)).

**Phase 2 — Map query data onto the reference**
- Use `map_newdata.py` to project new datasets onto the finished reference → arrow files, viewed as an overlay
  on the reference pyramid ([below](#view-mapped-query-data--map_newdatapy))

## Where to run

Two parts run on NRP, exchanging files with `<WORK>/<ATLAS_DIR>/` through `s3://latentbrain/<ATLAS_DIR>/`:

- KNN (Step 2): a CPU Job sized from the embedding file (500K cells: 16 cores, 32 GB RAM), launched from a RunKnnNRP
  clone on a machine with `kubectl` access to NRP: courtyard ([kubectl on courtyard](#kubectl-on-courtyard)) or
  your laptop. `run_build.sh` launches it by itself.
- PUMAP training (Step 3): a GPU pod running JupyterLab

Everything else (Step 1, PCA in Step 2, Steps 4–6) runs on courtyard (see
[below](#running-on-courtyard-gi-shared-machine)). Code and data live apart there:

- **Code: your own clones**, in your folder `/public/groups/hausslerlab/people/<login>/` (`<login>` = your courtyard
  user name): `FM_wrangling/`, `RunKnnNRP/`, `data-explorer/`. Each person has their own, so one person's `git pull` or
  edits never change another's runs. `datasetLists/` is inside `FM_wrangling/`; the scripts find it next to themselves.
- **Data: one shared folder**, `<WORK>` = `/public/groups/hausslerlab/projects/latentbrain/`, holding
  `<NPY_DIR>/` and `<DUP_DIR>/` (per-dataset embeddings and obs, filled by `getEmbeddingfromS3.py` in Step 1) and one
  `<ATLAS_DIR>/` per reference atlas (a name you choose). The atlas has a folder of the same name on S3,
  `s3://latentbrain/<ATLAS_DIR>/`: `run_build.sh`, the KNN Job and the PUMAP notebook all find their files by that
  name, so keep the two the same and do not rename either one.

Run every courtyard command from `<WORK>`, calling the scripts in your own clone. Set these once in `~/.bashrc`
(then `source ~/.bashrc`):
```bash
export FMW=/public/groups/hausslerlab/people/$USER/FM_wrangling
export WORK=/public/groups/hausslerlab/projects/latentbrain
```
then:
```bash
cd $WORK
python3 $FMW/combineEmbedding.py ...
```
The scripts do not read `$WORK`: they write relative to the folder you run them from, so `cd $WORK` first.
Files created in the shared folder are writable by the `hausslerlab` group automatically, with no `umask` change:
the folder has the setgid bit (new files get group `hausslerlab`) and a default ACL (new files are group-writable).

`<EMBEDDING>` below is the `--embedding` value of `getEmbeddingfromS3.py` and `combineEmbedding.py`: our name for
the foundation model the datasets were embedded with in
[RunFoundationModelNRP](https://github.com/ucscBrainExplorer/RunFoundationModelNRP). It sets where the embeddings
come from and which local folders hold them:

| `<EMBEDDING>` | Model | Dim | Reads from S3 | `<NPY_DIR>` | `<DUP_DIR>` |
|---|---|---|---|---|---|
| `uce` | UCE | 1280 | `s3://latentbrain/UCE/` | `UCE_npy_obs` | `duplicate_UCE` |
| `uce_brain_512` | UCE brain 512 | 512 | `s3://latentbrain/UCE_brain_512/` | `UCE_brain_512_npy_obs` | `duplicate_UCE_brain_512` |
| `uce_brain_768_bican` | UCE brain 768 + BICAN anneal | 768 | `s3://latentbrain/UCE_brain_768_bican/` | `UCE_brain_768_bican_npy_obs` | `duplicate_UCE_brain_768_bican` |
| `uce_brain_768_bican_pt` | the same + post-training aligning Siletti, BICAN BG, SEA-AD | 768 | `s3://latentbrain/UCE_brain_768_bican_pt/` | `UCE_brain_768_bican_pt_npy_obs` | `duplicate_UCE_brain_768_bican_pt` |
| `uce_brain_768_bican_s2048` | UCE brain 768 + BICAN anneal on 2,048-gene sentences | 768 | `s3://latentbrain/UCE_brain_768_bican_s2048/` | `UCE_brain_768_bican_s2048_npy_obs` | `duplicate_UCE_brain_768_bican_s2048` |


**S3 access.** `s3://latentbrain/<ATLAS_DIR>/` needs no setup: S3 has no real folders, so the first
`aws s3 cp <file> s3://latentbrain/<ATLAS_DIR>/` creates it (`aws s3 ls s3://latentbrain/<ATLAS_DIR>/` lists it). The
`aws` commands on courtyard need your own AWS key (ask Jing; enter it with `aws configure`, never in code or chat).
Keys can list, read and upload but not delete: ask Jing to delete files in S3. The NRP Jobs and pods use their own
key from the `latentbrain-aws-creds` secret.

---

## Steps 1–2 in one go on courtyard — `run_build.sh`

For several datasets, `run_build.sh` runs Step 1 (get and combine the embeddings) and Step 2: it downloads,
combines, uploads `uce.h5` to `s3://latentbrain/<ATLAS_DIR>/`, starts the KNN Job on NRP ([Step 2](#knn-nrp-cluster)),
runs IPCA while the KNN runs, and uploads `pca.h5`, `ipca_model.pkl` and `obs.tsv.gz`. A file already on S3 with the
same size is not uploaded again (e.g. a `uce.h5` uploaded by hand to start the KNN early). Copy it into the atlas
folder, set two things at its top, and run the copy in `screen`:
- `E`: the `<EMBEDDING>`
- `KNN_INDEX`: decide before starting, since the KNN Job starts by itself. `ivfflat` gives exact distances, but its
  index file is about the size of `uce.h5` (41 GB for 13.5M cells x 768). A PQ index (`ivfpq_<M>`) is compressed,
  with approximate distances, so fewer of the true nearest neighbours. `M` must divide the embedding size:
  768-d (UCE brain models): `ivfpq_256` (3 dims per code, about 12x smaller) or `ivfpq_128`;
  1280-d (UCE): `ivfpq_320` (4 dims per code, about 16x smaller). `none` skips the KNN.

It checks at the start that `kubectl` can create Jobs (see [kubectl on courtyard](#kubectl-on-courtyard)), so a
missing login fails at once rather than hours later.
```bash
cd $WORK && mkdir <ATLAS_DIR>
cp $FMW/run_build.sh <ATLAS_DIR>/     # list the dataset IDs, one per line, in <ATLAS_DIR>/dataset_ids.txt
cd <ATLAS_DIR> && bash run_build.sh 2>&1 | tee run_build.log
```
The atlas name comes from the folder the copy is in. The copy stays there as the record of how the atlas was built;
add per-atlas options to it if needed (e.g. `--obs-columns-file`). Steps 1 and 2 below explain each part.

---

## Step 1 — Get and combine embeddings (or shuffle a single dataset)

### Get the embeddings — `getEmbeddingfromS3.py`

Make `<ATLAS_DIR>/` and list its datasets in `<ATLAS_DIR>/dataset_ids.txt` (one dataset ID per line; lines starting
with `#` are skipped), then:
```bash
python3 $FMW/getEmbeddingfromS3.py --embedding <EMBEDDING> --select-datasets-file <ATLAS_DIR>/dataset_ids.txt
```

Downloads the embedded h5ads from `s3://latentbrain/` into `<NPY_DIR>/` as `<id>_uce.npy`, the obs table
and `<id>_provenance.json` (see [Provenance](#provenance-all-embeddings)). Datasets already there are skipped.

- Without `--select-datasets-file` it downloads every dataset under the embedding's S3 prefix, including other
  people's and ones still being embedded (e.g. all of HMBA, over 1 TB).
- It skips a dataset that looks unfinished, with a warning: a gap in its chunks, or a last chunk of full round size
  (a finished dataset's last chunk holds the remainder, e.g. `cell_70000_75374`). With `--select-datasets-file` it
  stops before downloading if any selected dataset is skipped this way or is not on S3.
- A dataset whose cell count really is a multiple of the chunk size looks unfinished too: check its `total_cells`
  and rerun with `--allow-full-last-chunk <id>`.

### Multiple datasets — use `combineEmbedding.py`

```bash
python3 $FMW/combineEmbedding.py \
    --embedding <EMBEDDING> \
    --results_dir <ATLAS_DIR>/ \
    --select-datasets-file <ATLAS_DIR>/dataset_ids.txt \
    --staging-dir /data/scratch/$USER \
    --shuffle
```
(`--staging-dir` on courtyard; see [Staging](#running-on-courtyard-gi-shared-machine).)

- `--select-datasets-file` — the datasets to combine, one dataset ID per line (matching the npy file names in
  `<NPY_DIR>/`); no header, lines starting with `#` are skipped. A line can add `<TAB>` and the exact source
  h5ad URL the embedding was run on (`s3://…` or `https://…`, with its release date, never "latest"), but this is
  only needed for datasets embedded before URLs were recorded: `RunFoundationModelNRP/fixed_pipeline.py` stores the
  URL in each output h5ad as `uns["source_url"]`, and `getEmbeddingfromS3.py` copies it into `<id>_provenance.json`.
  The script stops if a dataset has no URL either way, or if the two disagree.
- It writes `<ATLAS_DIR>/dataset_ids.tsv` with the URLs filled in (dataset ID `<TAB>` URL); if the select file is
  that same path, it is rewritten in place. `annotate_obs.py` reads the URLs for `metadata_summary.json`
  from it.
- `--shuffle` writes a new `shuffle_index.npy` and shuffles `uce.h5` and `obs.tsv.gz` in sync
- Verify output: cell count matches expectation, `obs.tsv.gz` contains expected columns

On startup the script validates that:
- All datasets in `<DUP_DIR>/` are listed in `datasetLists/ignore_datasets.txt`
- All `ucsc-*` prefixed datasets in `<NPY_DIR>/` and `<DUP_DIR>/` are listed in `datasetLists/ucsc_datasets.txt`

Run it from the shared folder (`/public/groups/hausslerlab/projects/latentbrain`, as `run_build.sh` does). It finds
`<NPY_DIR>/` relative to the current directory, and from anywhere else it stops with "no source URL" for every dataset.

### Datasets from several consortia: decide the obs columns first

When datasets come from different consortia (different column names and vocabularies), choose the combined columns
**before** the combine. Then `_full/` or the atlas is written once with harmonized, trimmed obs:

1. Profile each source obs in `<NPY_DIR>/<id>_obs.tsv`: column names, fill rate, values, and whether files from
   one consortium share a header.
2. Write the column list for `--obs-columns-file` and the `datasetLists/manual_column.json` rules
   ([Review obs](#review-obs-and-fill-in-manual_columnjson)). With many datasets or long value maps, generate the
   rules with a script kept in the atlas folder. The script builds each map from the values in the source obs and
   stops on any unmapped value. It is then also the record of how each column was made.
3. Keep the source column of every derived column in the list. A rule whose source column was dropped cannot be
   changed later with `annotate_obs.py`. Changing it would need an `--obs-only` re-combine plus
   `apply_shuffle_index.py`.
4. Test with `--obs-only` into a scratch `--results_dir`. It reads every source row through the rules and writes no
   embedding.

Rules are keyed by dataset ID and shared by every atlas that uses that ID. A rule that adds a new column to a dataset
also adds it to older atlases with that dataset the next time `annotate_obs.py` runs there, so prefer existing column
names where they do not collide. Example: `HMBA_SEAAD_PsychAD_ucebrain768bicans2048_shuffle/`
(`README.md`, `make_obs_rules.py`, `obs_columns.txt`).

### A subset of cells: combine into `_full/`, then filter rows

To keep only some cells, for example by assay or donor age:
1. Combine everything with `--shuffle` into `<ATLAS_DIR>/_full/`.
2. Run a small filter script that writes `<ATLAS_DIR>/uce.h5` and `obs.tsv.gz` from the kept rows, in the same
   order. Filtering a shuffled file keeps it shuffled, so no second shuffle is needed.
3. Copy `_full/{dataset_ids.tsv,dataset_provenance.json,wrangling_notes_combineEmbedding.txt}` up to `<ATLAS_DIR>/`.
4. Have the script write `wrangling_notes_custom.txt`, which `annotate_obs.py` adds to the wrangling notes.

Examples:
- `HNOCA_Extended_10x3v3_cell_768d/subset_10x3v3_cell.py`: assay and suspension type.
- `HMBA_SEAAD_PsychAD_ucebrain768bicans2048_shuffle/filter_psychad_adult.py`: PsychAD age ≥ 18. It streams obs line
  by line instead of loading it into pandas, and fills values the rules cannot make, such as a library ID cut from
  the cell ID.

### Single dataset — use `shuffle.py` directly

If only one dataset needs to be shuffled (no combine step needed):

```bash
python3 $FMW/utils/shuffle.py \
    --uce-in <NPY_DIR>/<dataset_id>_uce.npy \
    --obs-in <NPY_DIR>/<dataset_id>_obs.tsv \
    --out-dir <ATLAS_DIR>
```

Writes `<ATLAS_DIR>/uce.h5` (uncompressed), `obs.tsv.gz` and `shuffle_index.npy`. Obs is staged as Parquet in `--nvme-dir` (default `/nvme`) and values are kept verbatim.

Use `--chunked` if RAM is constrained: RAM drops to ~32 bytes/row plus one read block, but `--nvme-dir` needs free space equal to the full uncompressed array. `--chunked` requires `.h5` input; if the source h5 has tiny chunks (e.g. `combined_UCE_56M/uce.h5`), copy it to `/nvme` first and shuffle the copy.

## Step 2 — PCA and KNN (run in parallel)

### KNN (NRP cluster)

Copy `uce.h5` to S3 (on courtyard; `run_build.sh` does this):
```bash
aws s3 cp <ATLAS_DIR>/uce.h5 s3://latentbrain/<ATLAS_DIR>/uce.h5
```
then launch the job (`run_build.sh` does this too, with its `KNN_INDEX`) from your clone of
[RunKnnNRP](https://github.com/ucscBrainExplorer/RunKnnNRP) on a machine with `kubectl` access to NRP: courtyard
([kubectl on courtyard](#kubectl-on-courtyard)) or your laptop:
```bash
cd RunKnnNRP && git pull
./run_pipeline_knn_job.sh <ATLAS_DIR>                    # k=30, ivfflat
./run_pipeline_knn_job.sh <ATLAS_DIR> 30 ivfpq_256       # compressed index, for large atlases
./run_pipeline_knn_job.sh <ATLAS_DIR> 30 ivfflat ivfpq_128   # or several index types
kubectl get jobs -n braingeneers | grep knn              # progress
```

Each index type is one Kubernetes Job (`uce-job-knn-<dir>-k<K>-<index>`; a long `<dir>` is cut and ends with a short
hash of the full name) that reads `uce.h5` from
`s3://latentbrain/<ATLAS_DIR>/` and writes back next to it: `index_<INDEX_TYPE>.faiss`, `knn_<K>_<INDEX_TYPE>.npz`, and, for the first index type
listed, a copy named `index.faiss`. The production backend needs the atlas index as `index.faiss` or a file with
`ivfflat` in its name, so a PQ-only atlas relies on that copy.
PUMAP training reads the KNN file from S3, so it does not need to be copied to `<WORK>`.

### PCA

```bash
python3 $FMW/runIPCA.py --results_dir <ATLAS_DIR>
```

Runs IncrementalPCA (50 components) on `uce.h5` in streaming chunks. Options:
- `--n_components` — number of PCA components (default: 50)
- `--chunk_size` — rows per chunk (default: 100,000; set to 10,000 for small atlases <100k cells to avoid an h5py chunk shape error)
- `--group_name` — optional suffix for output filenames

Outputs: `ipca_model.pkl`, `pca.h5`. Copy them to S3 for PUMAP training, with `obs.tsv.gz` for its plots:
```bash
for f in pca.h5 ipca_model.pkl obs.tsv.gz; do aws s3 cp <ATLAS_DIR>/$f s3://latentbrain/<ATLAS_DIR>/$f; done
```

On a shared machine (e.g. courtyard), cap the BLAS threads — by default NumPy uses every core (a 7.1M × 512 run took ~34 of courtyard's 64 cores), and IncrementalPCA gains little beyond ~16:
```bash
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    python3 $FMW/runIPCA.py --results_dir <ATLAS_DIR>
```

## Step 3 — Train PUMAP

Runs on an NRP GPU pod in JupyterLab, with inputs and outputs in `s3://latentbrain/<ATLAS_DIR>/`: see
[train-pumap-nrp.md](train-pumap-nrp.md) for starting the pod. Open `3_trainPUMAPmodel.ipynb` and set
cell 2:
```python
results_dir = "s3://latentbrain/<ATLAS_DIR>/"
embedding_h5_file = "pca.h5"
knn_outfile = "knn_30_ivfflat.npz"
existing_model_iteration = 0   # 0 = start fresh; N = resume from pytorch{tag}_eN.pkl
epochs = 10                    # epochs to train (or add) in this run
k = 10                         # KNN neighbors used for PUMAP (subset of knn_30)
run_tag = ""                   # goes into every output name; use a new one for each try
```

The notebook trains PUMAP on the PCA-transformed embeddings using the precomputed KNN. Training is incremental — run multiple times to add epochs, saving a checkpoint each time (`pytorch_e{total}.pkl`). After training it transforms all cells and auto-detects if axes need swapping (width > height).

Outputs per run:
- `pytorch_e{N}.pkl` — PUMAP model checkpoint
- `transform_e{N}.json` — transform config used by `map_newdata.py`: the PUMAP model and ipca model file names (looked up next to the JSON, so keep them in the same folder), swap_xy_matrix, and for the record the training inputs `pca_file` and `knn_file`
- `umap_e{N}.npy` — 2D UMAP coordinates for all reference cells

(with a `run_tag`, e.g. `pytorch_k1_e{N}.pkl`). The last cell, `push(results_dir)`, uploads them to S3; copy the ones
you keep back to `<WORK>` for Step 5 and `map_newdata`:
```bash
for f in pytorch_e{N}.pkl transform_e{N}.json umap_e{N}.npy; do aws s3 cp s3://latentbrain/<ATLAS_DIR>/$f <ATLAS_DIR>/; done
```

## Step 4 — Annotate obs and write the metadata summary — `annotate_obs.py`

Usually done after PUMAP training, once the map looks right. It works on `<ATLAS_DIR>/obs.tsv.gz` from Step 1 and does
not touch the embeddings, so it can be rerun as often as needed.

### Review obs and fill in `manual_column.json`

Look through `obs.tsv.gz` (with Claude) for columns that are missing, empty or named differently in some datasets,
and add rules for them to `datasetLists/manual_column.json` in your FM_wrangling clone. Rules are keyed by dataset ID, then by the
obs column to fill. Three types, e.g.:

```json
"8e10f1c4-8e98-41e5-b65f-8cd89a887122": {
    "cell_type": {"type": "column", "value": "supercluster_term"},
    "organism":  {"type": "static", "value": "Homo sapiens"}
},
"ucsc-2e934674-0669-4d8a-9faa-9d37078f8bb6": {
    "cell_type_ontology_term_id": {"type": "map", "value": "Subclass",
                                   "map": {"Astrocyte": "CL:0000127", "ACx MEIS2 GABA": null}}
}
```

- `column`: copy another column of that dataset (here its cell types are in `supercluster_term`)
- `static`: one value for every cell of that dataset
- `map`: translate the values of a column (`value`) through `map`; `null` leaves it empty
- optional `"note"`: copied into the wrangling notes the site shows

The file is in the repo, so commit and push the change (the next atlas build uses the same rules).
`annotate_obs.py` applies the rules to `obs.tsv.gz` itself, so after adding one, just rerun it: no need to rerun
`combineEmbedding.py` or retrain.

### Run it

```bash
python3 $FMW/annotate_obs.py --data_dir <ATLAS_DIR>
```

- Reads `<ATLAS_DIR>/obs.tsv.gz` and `dataset_ids.tsv`, plus `datasetLists/` (`ucsc_datasets.txt`, `manual_column.json`, newest `datasets.info.cxg.*.txt` with `doi_overrides.txt` applied).
- Writes `obs_annotated.tsv.gz`, `metadata_summary.json`, `wrangling_notes_annotation.txt` and the two `system_prompt_*.txt` files.
- Normalizes organism and development stage (age bucket) with OpenAI gpt-4.1; needs `OPENAI_API_KEY` in a `.env` file, found by searching up from the script's folder: put it in
  `/public/groups/hausslerlab/people/<login>/.env`, readable only by you (`chmod 600`), never in the shared folder or
  inside the clone. The two mappings are printed for checking.
- Keeps the hand-filled fields of an existing `metadata_summary.json` (`reference_name`, `reference_uuid`, `version_label`, `reference_pyramid`, `foundation_model`, `abstract`); everything else is regenerated.
- `--cell_label <column>` sets the column whose labels are counted as the cell types in `metadata_summary.json`.
  Give the same column you pin first in the pyramid (`buildPyramidConfig.py --pinned`), so the site summary and the
  map agree. Without it, the first of `CELL_LABEL_PRIORITY` in `cell_label.py` is used, and the log says which.
  Example: `--cell_label concensus_subclass` for an HMBA atlas.
- `--output_dir <DIR>` writes the outputs elsewhere (still reading the hand-filled fields from `<ATLAS_DIR>`), to compare a re-run before replacing the files.

## Step 5 — Build pyramid

### Generate pyramid config

```bash
python3 $FMW/buildPyramidConfig.py \
    --results_dir <ATLAS_DIR> \
    --umap_file umap_e{N}.npy \
    --metadata_file obs_annotated.tsv.gz
```

Use the annotated obs from Step 4, which has the `normalized_*` columns; without `--metadata_file` the script reads
the unannotated `obs.tsv.gz`. The file name is saved in the config, so `buildPyramid.py` uses the same one.

This auto-generates `<ATLAS_DIR>/pyramid_e{N}_config.json` listing the obs columns to show, in this order:

1. **The cell label, pinned first:** the first of `harmonized_subclass`, `harmonized_cell_type`,
   `harmonized_cell_label`, `mapped_cell_label`, `concensus_subclass` (HMBA's cross-species subclass; only if it
   covers at least 90% of the cells `cell_type` covers, so not in a cohort where only the HMBA datasets have it),
   `custom_cell_type`,
   `subclass` (only if it covers more cells than `cell_type`), `cell_type` present in the obs. The list is `CELL_LABEL_PRIORITY` in `cell_label.py`, which
   `annotate_obs.py` also uses for the cell-type counts in `metadata_summary.json`, so the map and the site's summary
   use the same column. Check which one was picked in the printed summary; `--pinned <column>` overrides it.
   `--pinned` also takes a comma-separated list, pinned first in that order, the first being the cell label: for
   example `--pinned concensus_subclass,species` for a cross-species atlas.
2. **Every other column, in one alphabetical list, ignoring case** (so `Class` sits between `chemistry` and
   `cluster`, not at the top). The list includes:
   - **HIGH_PRIORITY columns**, always: `HIGH_PRIORITY` in `buildPyramidConfig.py`, which is always `sample_label`
     and `donor_id`, plus for example `assay`, `disease`, `tissue` and `normalized_organism`. They are included
     regardless of null rate or category count, but are not listed first; a separate block made columns hard to
     find (changed 2026-11-07).
   - **Every other eligible column:** at most 50% null (within at least one collection) and, for text columns, fewer
     categories than a tenth of the cells.

`--columns a,b,c` limits the config to those columns.

Fill in before building the tiles:
- `reference_name` — human-readable name
- `target_dir` — where tiles will be written
- These can be copied from a prior config if rebuilding

### Clear old target directory

Clear everything in `target_dir` — PNGs, arrow files, and metadata.json all belong to the old reference and are no longer valid:
```bash
find <target_dir> -mindepth 1 -delete
```

### Build the tiles

```bash
python3 $FMW/buildPyramid.py \
    --results_dir <ATLAS_DIR> \
    --pyramid_file pyramid_e{N}_config.json
```

The script auto-selects tile resolution to keep pixel loss rate <3%, builds multi-level PNG tiles for each feature, and writes `metadata.json` to `target_dir` (a relative `target_dir` is taken under `<ATLAS_DIR>`).

Options:
- `--skip_top_levels N` — don't write tiles for the N highest-resolution levels (`metadata.json` still lists all levels)
- `--metadata_only` — write `metadata.json` without building tiles

Output: PNG tiles for each feature + `metadata.json` in `target_dir`. The tile count grows with the number of levels
(e.g. 525 per feature at 7 levels).

### Upload and view

Upload `target_dir` to a new folder in the pyramid bucket, named `<ATLAS_DIR>_<reference_uuid>`. The top level of the
bucket is for production atlases (the ones data-explorer shows); put experimental ones under `experimental/`, and copy
one to the top level only when it becomes production. `<PREFIX>` below is `experimental/` or empty:
```bash
aws s3 ls s3://latentbrainpyramid/<PREFIX><ATLAS_DIR>_<reference_uuid>/        # must be empty: a new folder
aws s3 sync --no-progress <target_dir> s3://latentbrainpyramid/<PREFIX><ATLAS_DIR>_<reference_uuid>/
```
Its URL is `https://latentbrainpyramid.s3.amazonaws.com/<PREFIX><ATLAS_DIR>_<reference_uuid>` (a plain `curl` of it
gets 403; that is normal). View it in the browser, before or without adding it to data-explorer:
```
https://brainexplorer.ucsc.edu/pyramid/?image=https://latentbrainpyramid.s3.amazonaws.com/<PREFIX><ATLAS_DIR>_<reference_uuid>
```

## Step 6 — Add the reference to data-explorer — `merge_metadata_summary.py`

Set `reference_pyramid` in `<ATLAS_DIR>/metadata_summary.json` to the pyramid's public URL, then merge it into the data-explorer metadata list:

```bash
python3 $FMW/merge_metadata_summary.py /public/groups/hausslerlab/people/$USER/data-explorer/UCSC_Brain_Explorer_metadata.json <ATLAS_DIR>/metadata_summary.json
```

- Matches entries by `reference_uuid`: an existing uuid is replaced whole (each field change is printed), a new uuid is appended.
- Writes a timestamped `.bak` of the target next to it; delete it once the diff looks right.

Copy the notes the site shows into `wrangling_notes/<reference_uuid>/` in your data-explorer clone:
- `<ATLAS_DIR>/wrangling_notes_annotation.txt` → `wrangling_notes.txt`
- `system_prompt_normalized_age_bucket.txt` and `system_prompt_normalized_organism.txt`

## View mapped query data — `map_newdata.py`

The Arrow file from `map_newdata.py --out <name>.arrow` shows as an overlay on the reference. Upload it into a
`query/` folder inside the reference's pyramid folder, and add `&overlay=` to the viewer URL:
```bash
aws s3 cp <ATLAS_DIR>/query/<name>.arrow s3://latentbrainpyramid/<ATLAS_DIR>_<reference_uuid>/query/<name>.arrow
```
The overlay's coordinates come from the PUMAP that pyramid was drawn from, so it belongs with that pyramid; a new
PUMAP means a new pyramid, and the overlay has to be made again. Re-uploading the pyramid (`aws s3 sync`, without
`--delete`) leaves the `query/` folder in place.
Not `s3://latentbrain/`: the viewer cannot read that bucket (403). The pyramid bucket serves only requests coming
from the brainexplorer site, so opening the file's URL directly also gives 403; to check an upload, send the site as
Referer: `curl -sI -H "Referer: https://brainexplorer.ucsc.edu/" <arrow_url> | head -1` should print `200 OK`.
```
{browser}/pyramid/?image={pyramid_url}&overlay={arrow_url}
```
- `{browser}`: `https://brainexplorer.ucsc.edu`, or `https://dev.brainexplorer.ucsc.edu` for the dev site
- `{pyramid_url}`: the reference pyramid's public URL (Step 5, Upload and view)
- `{arrow_url}`: the Arrow file's URL, `https://latentbrainpyramid.s3.amazonaws.com/<ATLAS_DIR>_<reference_uuid>/query/<name>.arrow`

For example, the Phan 2024 striatum (c893ddc3) on HMBA v0.5:
`image=https://latentbrainpyramid.s3.amazonaws.com/HMBAv0.5_ucebrain768bicans2048_shuffle_b62573e1-2880-45ca-833c-89fa61e751e2`
with `overlay=https://latentbrainpyramid.s3.amazonaws.com/HMBAv0.5_ucebrain768bicans2048_shuffle_b62573e1-2880-45ca-833c-89fa61e751e2/query/c893ddc3-f25b-45e2-8c9e-155918b4261c.arrow`.

---

## Running on courtyard (GI shared machine)

- **Shared folder rules** (`<WORK>` = `/public/groups/hausslerlab/projects/latentbrain/`, see
  [Where to run](#where-to-run)):
  - New files there are group-writable automatically (setgid + default ACL). A file copied in with `cp -p` or
    `rsync -a` keeps its old permissions: if someone can't edit it, run `chmod g+w` on it.
  - `git pull` your FM_wrangling clone before `combineEmbedding.py` or `annotate_obs.py`, so everyone uses the same
    `datasetLists/` (`manual_column.json`, `ignore_datasets.txt`, ...).
  - When an atlas is finished, make it read-only: `chmod -R g-w <ATLAS_DIR>`.
  - Your folder under `people/` is readable by everyone by default: keep files you don't mean to share (`.env`, keys)
    private with `chmod 600`.
- **Home directory:** don't write large outputs there: it has a ~31 GB quota, and going over fails with "Disk quota
  exceeded".
- **Staging (scratch space):** `combineEmbedding.py` first writes its big outputs (`uce.h5`, `obs.tsv.gz`, and with
  `--shuffle` a temporary copy of the whole embedding matrix) to a fast local disk, then moves them into
  `<ATLAS_DIR>/`. That disk is `--staging-dir`, default `/nvme` (the AWS machine's local SSD), which courtyard does not
  have. Use your own folder on courtyard's local scratch disk instead: `mkdir -p /data/scratch/$USER` once, then add
  `--staging-dir /data/scratch/$USER` to every `combineEmbedding.py` command (and `--nvme-dir /data/scratch/$USER` to
  `utils/shuffle.py`). It needs free space about the size of the output `uce.h5`. Files there not accessed for 14 days
  are deleted automatically, which is fine: they are only temporary.
- **Threads:** cap NumPy/sklearn jobs with `OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16` (see PCA above).
- **No AVX:** the CPU (Xeon E7-4830) cannot run `tiledbsoma` / `cellxgene_census` ("Illegal instruction"). `combineEmbedding.py` reads CELLxGENE metadata from the newest `datasetLists/datasets.info.cxg.*.txt` instead; refresh it with `$FMW/datasetLists/fetch_cxg_datasets.py` (writes into that folder by default). Where CELLxGENE still cites a preprint that has since been published, add a row to `datasetLists/doi_overrides.txt` (dataset_id, collection_doi, collection_doi_label, note); both `combineEmbedding.py` and `annotate_obs.py` apply it on top of the table, and print how many overrides they used.
- **Long jobs:** start them in `screen` so they survive the terminal closing.

### kubectl on courtyard

NRP Jobs (the KNN) can be launched from courtyard with the lab's shared NRP service account (`braingeneers-ingest`,
namespace `braingeneers`), which needs no browser login. Once per person:
1. Get the kubeconfig file `config.service_account` from the lab (Slack) and copy it to courtyard, readable only by you
   (it holds a token, like a password; keep it out of `<WORK>`, repos and chat):
   ```bash
   scp config.service_account courtyard:~/.kube/        # from your laptop; mkdir -p ~/.kube first
   chmod 700 ~/.kube && chmod 600 ~/.kube/config.service_account
   ```
2. Install `kubectl` in `~/bin`, matching the cluster's minor version (v1.34 in 2026-11; `kubectl version` shows
   both, and warns if they are more than one apart):
   ```bash
   V=$(curl -fsSL https://dl.k8s.io/release/stable-1.34.txt)
   curl -fsSLO https://dl.k8s.io/release/$V/bin/linux/amd64/kubectl
   echo "$(curl -fsSL https://dl.k8s.io/release/$V/bin/linux/amd64/kubectl.sha256)  kubectl" | sha256sum --check
   install -m 755 kubectl ~/bin/kubectl && rm kubectl
   ```
3. Add to `~/.bashrc`, then `source ~/.bashrc`:
   ```bash
   export KUBECONFIG=$HOME/.kube/config.service_account
   ```
4. Check: `kubectl auth can-i create jobs -n braingeneers` prints `yes`.

Every command also prints a warning about "auto-generated secret-based tokens": harmless, it is about how the
service account was set up. No `kubectl oidc-login` plugin is needed (that is for personal browser logins). The
service account can also read the namespace's secrets, including `latentbrain-aws-creds`, so treat the file as a
credential.

### UCE brain 512 (`--embedding uce_brain_512`)

- Inputs come from `UCE_brain_512_npy_obs/`; embeddings are 512-dim instead of 1280.
- Example: `combined_UCE_5neuro_ucebrain512_shuffle_7253ea1e-e5ff-4f93-80fa-c108806fc7c6/` (5 brain datasets, 7.1M cells, ~16 min with `--shuffle`):
  ```bash
  python3 $FMW/combineEmbedding.py --embedding uce_brain_512 \
      --results_dir combined_UCE_5neuro_ucebrain512_shuffle_7253ea1e-e5ff-4f93-80fa-c108806fc7c6 \
      --select-datasets-file combined_UCE_5neuro_ucebrain512_shuffle_7253ea1e-e5ff-4f93-80fa-c108806fc7c6/dataset_ids.tsv \
      --obs-columns-file combined_UCE_5neuro_ucebrain512_shuffle_7253ea1e-e5ff-4f93-80fa-c108806fc7c6/obs_columns_brain.txt \
      --staging-dir /data/scratch/$USER --shuffle
  ```
- `--obs-columns-file` (select mode only) limits obs to the listed columns, in file order; without it, select mode writes the union of all columns.
- `datasetLists/manual_column.json` applies in both modes; rule types `static`, `column` and `map` (value translation, `null` = empty), each with an optional `note` that is copied into `wrangling_notes_combineEmbedding.txt`.
- KNN: use `--index_type ivfflat` or `ivfpq_128`; `ivfpq_320` fails because 512 is not divisible by 320.
- PCA/PUMAP models trained on 1280-dim UCE do not apply; retrain on the 512-dim `uce.h5`, and map only query data embedded with the same UCE brain model.

### UCE brain 768 BICAN (`--embedding uce_brain_768_bican`)

- Same flow as UCE brain 512, for the `ucebrain768bican` model (UCE-brain d768l14v2 + BICAN anneal) in RunFoundationModelNRP.
- `getEmbeddingfromS3.py --embedding uce_brain_768_bican` reads `s3://latentbrain/UCE_brain_768_bican/*_ucebrain768bican_adata.h5ad` (obsm `X_ucebrain768bican`) into `UCE_brain_768_bican_npy_obs/`; `combineEmbedding.py` writes a 768-dim `uce.h5`.
- KNN: 768 is divisible by 128 and 256 but not 320, so `ivfpq_320` fails here too.

### UCE brain 768 BICAN post-trained (`--embedding uce_brain_768_bican_pt`)

- Same flow as UCE brain 768 BICAN, for the `ucebrain768bicanpt` model (`ucebrain768bican` + post-training steps; see `submit_jobs.py` in RunFoundationModelNRP).
- `getEmbeddingfromS3.py --embedding uce_brain_768_bican_pt` reads `s3://latentbrain/UCE_brain_768_bican_pt/*_ucebrain768bicanpt_adata.h5ad` (obsm `X_ucebrain768bicanpt`) into `UCE_brain_768_bican_pt_npy_obs/`.
- Its embeddings differ from `ucebrain768bican`: do not combine the two or reuse that model's PCA, KNN, species directions or PUMAP.

### UCE brain 768 BICAN s2048 (`--embedding uce_brain_768_bican_s2048`)

- Same flow as UCE brain 768 BICAN, for the `ucebrain768bicans2048` model (Kuan's d768l14v2 BICAN anneal trained on 2,048-gene cell sentences instead of 1,024; see `submit_jobs.py` in RunFoundationModelNRP).
- `getEmbeddingfromS3.py --embedding uce_brain_768_bican_s2048` reads `s3://latentbrain/UCE_brain_768_bican_s2048/*_ucebrain768bicans2048_adata.h5ad` (obsm `X_ucebrain768bicans2048`) into `UCE_brain_768_bican_s2048_npy_obs/`.
- A separate model: do not combine its embeddings with any other model's or reuse their PCA, KNN, species directions or PUMAP.

### Provenance (all embeddings)

- `getEmbeddingfromS3.py` writes `<id>_provenance.json` (source h5ad URL, `source_verified`, pipeline commits, bf16/fp32 chunk counts, and the model's `uns["<method>_provenance"]`, e.g. uce-brain commit) next to each `.npy`.
- It stops on a chunk with `source_verified = False` (embedded from a staged copy with no record of where it came from), or whose source URL or model provenance differs from the dataset's other chunks. bf16 may differ (it follows the GPU; bf16 and fp32 embeddings are almost identical). Chunks from before these fields existed pass.
- It also checks every chunk for NaN embedding rows. A chunk with more than 1% NaN rows is a failed embedding run (the uce-brain 768 build had 8 chunks 61-90% NaN): it is listed under `nan_chunks` in `_provenance.json`, and the run exits 1 at the end. Datasets with `nan_chunks` are downloaded again on the next run, so re-embed the chunks upstream and rerun. `combineEmbedding.py` (select mode) stops before combining if any selected dataset has `nan_chunks`.
- Chunks with 1% or fewer NaN rows are only printed. `combineEmbedding.py` drops those cells, logs how many per dataset, and checks that cells written + NaN-dropped equals the pre-scan count.
- In select mode `combineEmbedding.py` collects them into `<results_dir>/dataset_provenance.json` (null for datasets with no `_provenance.json`) and warns if datasets were embedded with different model provenance.

## Reference

- Scripts: this repo — `combineEmbedding.py`, `annotate_obs.py`, `runIPCA.py`, `3_trainPUMAPmodel.ipynb`,
  `buildPyramidConfig.py`, `buildPyramid.py`; query mapping: `map_newdata.py`
- KNN: [RunKnnNRP](https://github.com/ucscBrainExplorer/RunKnnNRP); embeddings: [RunFoundationModelNRP](https://github.com/ucscBrainExplorer/RunFoundationModelNRP)
