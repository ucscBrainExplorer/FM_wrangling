# FM_wrangling

Builds browsable reference atlases for [UCSC Brain Explorer](https://brainexplorer.ucsc.edu) from
foundation-model embeddings of single-cell data, and maps new datasets onto them.

The embeddings come from [RunFoundationModelNRP](https://github.com/ucscBrainExplorer/RunFoundationModelNRP)
(UCE and the UCE-brain models, run on NRP). From those, this repo combines datasets, runs PCA, trains a
parametric UMAP (PUMAP), annotates cell metadata, and builds the image pyramid the browser shows.
`map_newdata.py` then projects a new dataset onto a finished atlas as an overlay.

## Docs

| Doc | What it covers |
|---|---|
| [build-reference.md](build-reference.md) | The whole pipeline, step by step: combine embeddings, PCA and KNN, PUMAP, annotate, pyramid, data-explorer, and mapping new data |
| [train-pumap-nrp.md](train-pumap-nrp.md) | Step 3 in detail: training PUMAP in JupyterLab on an NRP GPU pod |

## Where things run

| Part | Where |
|---|---|
| Combine embeddings, PCA, annotate, pyramid (Steps 1, 2, 4–6) | courtyard, from the shared folder `/public/groups/hausslerlab/projects/latentbrain/` |
| KNN (Step 2) | an NRP CPU Job, launched with [RunKnnNRP](https://github.com/ucscBrainExplorer/RunKnnNRP) |
| PUMAP training (Step 3) | an NRP GPU pod ([train-pumap-nrp.md](train-pumap-nrp.md)) |
| Launching NRP Jobs and pods | any machine with `kubectl` access to the `braingeneers` namespace: courtyard (shared service account, see [build-reference.md](build-reference.md#kubectl-on-courtyard)) or your laptop |

Files move between courtyard and NRP through `s3://latentbrain/<ATLAS_DIR>/`. See
[build-reference.md](build-reference.md#where-to-run) for the folder layout and setup.
