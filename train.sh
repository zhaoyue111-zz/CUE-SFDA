#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

cue_python=/home/ubuntu/anaconda3/envs/voxtell/bin/python
cue_task=${1:-all}
cue_liver_gpu=${CUE_LIVER_GPU:-0}
cue_cancer_gpu=${CUE_CANCER_GPU:-1}

train_liver() {
  "$cue_python" scripts/run_cue_sfda.py train \
    --config configs/dwi_liver.yaml \
    --gt-path /data/hilab/datasets/WCH_MRI145/DWI/liver \
    --output-dir outputs/dwi_only_liver_v2 \
    --iterations 3000 \
    --gpu-id "$cue_liver_gpu" \
    --resume outputs/dwi_only_liver_v2/checkpoints/latest.pt

#  "$cue_python" scripts/run_cue_sfda.py evaluate \
#    --config configs/dwi_liver.yaml \
#    --checkpoint outputs/dwi_only_liver/checkpoints/latest.pt \
#    --output-dir outputs/dwi_only_liver/evaluation \
#    --gt-path /data/hilab/datasets/WCH_MRI145/DWI/liver \
#    --gpu-id "$cue_liver_gpu"
}

train_cancer() {
  "$cue_python" scripts/run_cue_sfda.py train \
    --config configs/dwi_liver_cancer.yaml \
    --output-dir outputs/dwi_only_liver_cancer_v2 \
    --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels \
    --iterations 1000 \
    --gpu-id "$cue_cancer_gpu" \
    --resume outputs/dwi_only_liver_cancer_v2/checkpoints/latest.pt

#  "$cue_python" scripts/run_cue_sfda.py evaluate \
#    --config configs/dwi_liver_cancer.yaml \
#    --checkpoint outputs/dwi_only_liver_cancer/checkpoints/latest.pt \
#    --output-dir outputs/dwi_only_liver_cancer/evaluation \
#    --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels \
#    --gpu-id "$cue_cancer_gpu"
}

case "$cue_task" in
  liver)
    train_liver
    ;;
  cancer)
    train_cancer
    ;;
  all)
    train_liver &
    cue_liver_pid=$!
    train_cancer &
    cue_cancer_pid=$!
    cue_status=0
    wait "$cue_liver_pid" || cue_status=$?
    wait "$cue_cancer_pid" || cue_status=$?
    exit "$cue_status"
    ;;
  *)
    echo "Usage: $0 [liver|cancer|all]" >&2
    exit 2
    ;;
esac
