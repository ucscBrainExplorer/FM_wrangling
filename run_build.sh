#!/bin/bash
# Steps 1-2 of build-reference.md on courtyard, for one reference atlas:
#   download the listed datasets' embeddings, combine (shuffled, obs = union of columns), upload uce.h5 and
#   start the KNN Job on NRP (RunKnnNRP), then IPCA and upload pca.h5, ipca_model.pkl, obs.tsv.gz
#   to s3://latentbrain/<ATLAS_DIR>/. The KNN runs on NRP while the IPCA runs here; PUMAP (NRP GPU pod) comes after.
#
# Copy this file into the atlas folder, set E and KNN_INDEX below, and run the copy there (in screen):
#   cd $WORK && mkdir <ATLAS_DIR>
#   cp $FMW/run_build.sh <ATLAS_DIR>/          # list the dataset IDs, one per line, in <ATLAS_DIR>/dataset_ids.txt
#   cd <ATLAS_DIR> && bash run_build.sh 2>&1 | tee run_build.log
# The copy stays in the atlas folder as the record of how it was built; add per-atlas options there if needed.

E=CHANGE_ME          # the --embedding value, e.g. uce_brain_768_bican_s2048 (see the table in build-reference.md)
KNN_INDEX=CHANGE_ME  # FAISS index for the KNN Job: ivfflat (exact distances; index file about the size of uce.h5),
                     # or a compressed PQ index (approximate distances): 768-d ivfpq_256 (or ivfpq_128),
                     # 1280-d ivfpq_320 (the number must divide the embedding size); or none to start the KNN yourself.

set -euo pipefail
FMW=${FMW:-/public/groups/hausslerlab/people/$USER/FM_wrangling}
RUNKNN=${RUNKNN:-/public/groups/hausslerlab/people/$USER/RunKnnNRP}   # your RunKnnNRP clone
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$(basename "$HERE")           # the atlas folder is the one this copy sits in; the S3 folder has the same name
cd "$(dirname "$HERE")"         # the shared folder: the scripts write <NPY_DIR>/ and <ATLAS_DIR>/ relative to it
if [ "$E" = CHANGE_ME ] || [ "$KNN_INDEX" = CHANGE_ME ]; then
    echo "set E and KNN_INDEX at the top of $HERE/run_build.sh" >&2; exit 1
fi
if [ "$HERE" = "$(cd "$FMW" && pwd)" ]; then
    echo "copy run_build.sh into the atlas folder and run the copy (see the top of the file)" >&2; exit 1
fi
if [ ! -f "$D/dataset_ids.txt" ]; then
    echo "missing $HERE/dataset_ids.txt (one dataset ID per line)" >&2; exit 1
fi
# Check NRP access now, not hours later when the KNN is due to start.
if [ "$KNN_INDEX" != none ]; then
    if [ ! -x "$RUNKNN/run_pipeline_knn_job.sh" ]; then
        echo "no RunKnnNRP clone at $RUNKNN (set RUNKNN, or KNN_INDEX=none)" >&2; exit 1
    fi
    if ! kubectl auth can-i create jobs -n braingeneers 2>/dev/null | grep -qx yes; then
        echo "kubectl cannot create Jobs in braingeneers (KUBECONFIG set? see build-reference.md), or set KNN_INDEX=none" >&2
        exit 1
    fi
fi
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16   # shared courtyard
mkdir -p /data/scratch/$USER
echo "== $E -> $HERE, KNN $KNN_INDEX, FM_wrangling $(git -C "$FMW" log -1 --format=%h)$(git -C "$FMW" status --porcelain | grep -q . && echo ' (with uncommitted changes)' || true)"

# Upload unless S3 already has a file of the same size (e.g. uploaded by hand to start the KNN early).
upload() {
    local f=$1 s3="s3://latentbrain/$D/$1" remote
    remote=$(aws s3 ls "$s3" 2>/dev/null | awk -v n="$1" '$4 == n {print $3}' || true)
    if [ "$remote" = "$(stat -c %s "$D/$f")" ]; then
        echo "   $f: already on S3 with the same size, not uploaded again"
    else
        aws s3 cp --no-progress "$D/$f" "$s3"
    fi
}

echo "== download start $(date)"
python3 -u "$FMW/getEmbeddingfromS3.py" --embedding "$E" --select-datasets-file "$D/dataset_ids.txt"

echo "== combine start $(date)"
python3 -u "$FMW/combineEmbedding.py" --embedding "$E" \
    --results_dir "$D" --select-datasets-file "$D/dataset_ids.txt" \
    --staging-dir /data/scratch/$USER --shuffle

echo "== upload uce.h5 $(date)"
upload uce.h5

if [ "$KNN_INDEX" != none ]; then
    echo "== start KNN Job on NRP ($KNN_INDEX) $(date)"
    # drop kubectl's service-account token warning; a failed launch still stops the script (pipefail)
    (cd "$RUNKNN" && ./run_pipeline_knn_job.sh "$D" 30 "$KNN_INDEX") 2>&1 | { grep -v 'auto-generated secret-based tokens' || true; }
fi

echo "== ipca start $(date)"
python3 -u "$FMW/runIPCA.py" --results_dir "$D"

echo "== upload pca.h5, ipca_model.pkl, obs.tsv.gz $(date)"
for f in pca.h5 ipca_model.pkl obs.tsv.gz; do upload "$f"; done

echo "== done $(date)"
if [ "$KNN_INDEX" = none ]; then
    echo "   Next, the KNN from your RunKnnNRP clone: ./run_pipeline_knn_job.sh $D"
else
    echo "   KNN Job running on NRP; check with: kubectl get jobs -n braingeneers | grep knn"
fi
