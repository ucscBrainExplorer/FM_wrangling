# Train PUMAP on an NRP GPU pod

`3_trainPUMAPmodel.ipynb` and `3_trainPUMAPmodel_supervised.ipynb` run in JupyterLab on an
NRP GPU, reading their inputs from S3 and writing results back to S3. The pod runs a copy of
`main` cloned when it starts. See [build-reference.md](build-reference.md) Step 3 for what the
notebook does.

You need `kubectl` access to the NRP namespace `braingeneers`.

## 1. Put the inputs in S3

One folder per run, e.g. `s3://latentbrain/combined_subsample_500K_ucebrain768bicanpt_shuffle/`:

| File | Used by |
|---|---|
| `pca.h5` (or `pca_corrected*.h5`) | training (`embedding_h5_file`) |
| `knn_30_ivfflat.npz` | training (`knn_outfile`) |
| `obs.tsv.gz` | plot cells (supervised: `metadata_file`, the labels) |
| `ipca_model.pkl` | not training; `map_newdata` needs it next to the transform JSON |

## 2. Start the pod

Start it first: it may wait for a GPU, and the first start then takes 5–6 minutes to install.
Use your own name, so two people can each have a pod:

```bash
sed s/USERNAME/chris/g k8s/jupyter-pumap-job.yaml | kubectl apply -f -

kubectl get pods -n braingeneers -l job-name=jupyter-pumap-chris    # Pending until a GPU frees up
```

The pod clones a fresh copy of `main`, installs the pinned packages
([requirements_pumap.txt](requirements_pumap.txt), the AWS GPU instance's versions) and starts
JupyterLab. Later starts reuse a pip cache.

## 3. Connect

```bash
kubectl logs -n braingeneers job/jupyter-pumap-chris | grep "Jupyter token"   # after the install finishes

kubectl port-forward -n braingeneers job/jupyter-pumap-chris 8888:8888    # keep open
```

Open http://localhost:8888 and paste the token. If the laptop sleeps, rerun `port-forward`; if
the pod restarted, get the new token.

## 4. Set cell 2 and run

Open the notebook and, in cell 2, set `results_dir` to the S3 folder and adjust the settings:

```python
results_dir = "s3://latentbrain/combined_subsample_500K_ucebrain768bicanpt_shuffle/"
embedding_h5_file = "pca.h5"
existing_model_iteration = 0   # 0 = new model; N = continue from pytorch{tag}_eN.pkl
epochs = 10
run_tag = "try1"               # new tag for each try; goes into every output name
```

Then Run All and look at the maps. Edits made in the pod are scratch: they vanish with it (the
outputs in S3 stay). To keep a change, make it on your laptop and `git push`; to bring a pushed
change into a running pod, run `git pull` in a JupyterLab terminal.

At the end of cell 2, `workdir()` ([s3dir.py](s3dir.py)) copies just the files training
reads into the pod; the last cell, `push(results_dir)`, uploads the new outputs. Neither
ever overwrites a file: if an output with the same name is already in S3, `push` warns and
skips it, so use a new `run_tag` (or more epochs) for each try. `push` uploads only files
made since the kernel started, so a file you delete from S3 is not re-uploaded from the
pod's cache on `/mnt/pvc` (restarting the kernel mid-run makes earlier outputs count as
old). A local path such as
`/mnt/efsUCE/...` still works on the AWS GPU instance as before.

## 5. Delete the pod when done

NRP flags GPUs that sit idle:

```bash
kubectl delete job -n braingeneers jupyter-pumap-chris
```

Results are in the S3 folder (`pytorch{tag}_e{N}.pkl`, `transform{tag}_e{N}.json`,
`umap{tag}_e{N}.npy`). Downloaded inputs stay cached on the shared PVC (`uce-pvc`, folder
`pumap/<prefix>/`), so the next pod skips the download.
