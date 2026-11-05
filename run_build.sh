#!/bin/bash
# Steps 1-2 of build-reference.md on courtyard, for one reference atlas:
#   download the listed datasets' embeddings, combine (shuffled, obs = union of columns), IPCA, and upload
#   uce.h5 (for the KNN) and pca.h5, ipca_model.pkl, obs.tsv.gz (for PUMAP) to s3://latentbrain/<ATLAS_DIR>/.
# The KNN (RunKnnNRP) and PUMAP (NRP GPU pod) run after this, on NRP, launched from a machine with kubectl.
#
# Copy this file into the atlas folder, set E below, and run the copy there (in screen):
#   cd $WORK && mkdir <ATLAS_DIR>
#   cp $FMW/run_build.sh <ATLAS_DIR>/          # list the dataset IDs, one per line, in <ATLAS_DIR>/dataset_ids.txt
#   cd <ATLAS_DIR> && bash run_build.sh 2>&1 | tee run_build.log
# The copy stays in the atlas folder as the record of how it was built; add per-atlas options there if needed.

E=CHANGE_ME     # the --embedding value, e.g. uce_brain_768_bican_s2048 (see the table in build-reference.md)

set -euo pipefail
FMW=${FMW:-/public/groups/hausslerlab/people/$USER/FM_wrangling}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$(basename "$HERE")           # the atlas folder is the one this copy sits in
cd "$(dirname "$HERE")"         # the shared folder: the scripts write <NPY_DIR>/ and <ATLAS_DIR>/ relative to it
if [ "$E" = CHANGE_ME ]; then
    echo "set E at the top of $HERE/run_build.sh" >&2; exit 1
fi
if [ "$HERE" = "$(cd "$FMW" && pwd)" ]; then
    echo "copy run_build.sh into the atlas folder and run the copy (see the top of the file)" >&2; exit 1
fi
if [ ! -f "$D/dataset_ids.txt" ]; then
    echo "missing $HERE/dataset_ids.txt (one dataset ID per line)" >&2; exit 1
fi
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16   # shared courtyard
mkdir -p /data/scratch/$USER
echo "== $E -> $HERE, FM_wrangling $(git -C "$FMW" log -1 --format=%h)$(git -C "$FMW" status --porcelain | grep -q . && echo ' (with uncommitted changes)' || true)"

echo "== download start $(date)"
python3 -u "$FMW/getEmbeddingfromS3.py" --embedding "$E" --select-datasets-file "$D/dataset_ids.txt"

echo "== combine start $(date)"
python3 -u "$FMW/combineEmbedding.py" --embedding "$E" \
    --results_dir "$D" --select-datasets-file "$D/dataset_ids.txt" \
    --staging-dir /data/scratch/$USER --shuffle

echo "== upload uce.h5 $(date)"
aws s3 cp --no-progress "$D/uce.h5" "s3://latentbrain/$D/uce.h5"

echo "== ipca start $(date)"
python3 -u "$FMW/runIPCA.py" --results_dir "$D"

echo "== upload pca.h5, ipca_model.pkl, obs.tsv.gz $(date)"
for f in pca.h5 ipca_model.pkl obs.tsv.gz; do aws s3 cp --no-progress "$D/$f" "s3://latentbrain/$D/$f"; done

echo "== done $(date). Next, the KNN from your RunKnnNRP clone on a machine with kubectl (e.g. your laptop):"
echo "   ./run_pipeline_knn_job.sh $D"
