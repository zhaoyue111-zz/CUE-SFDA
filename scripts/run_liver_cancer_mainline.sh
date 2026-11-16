#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

cue_python=${CUE_PYTHON:-/home/ubuntu/anaconda3/envs/voxtell/bin/python}
gpu=${CUE_GPU:-0}
config=configs/response_routing_liver_cancer.yaml
gt_path=/data/hilab/datasets/WCH_MRI145/DWI/labels
root=outputs/liver_cancer_mainline

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 METHOD CONSENSUS_MODE" >&2
  echo "Example: $0 A_C_baseline none" >&2
  exit 2
fi

method=$1
mode=$2
output_dir="$root/$method"
source_cache=outputs/ablation_consensus_cache
consensus_cache="$output_dir/consensus_cache"
if [[ "$mode" == "joint_language" ]]; then
  consensus_cache="$source_cache"
elif [[ "$mode" == "dynamic_hard_language" || "$mode" == "selective_dynamic" || "$mode" == "selective_dynamic_language" ]]; then
  if [[ ! -f "$consensus_cache/metadata.json" ]]; then
    "$cue_python" - "$source_cache" "$consensus_cache" <<'PY'
import os
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
(target / "cases").mkdir(parents=True, exist_ok=True)
for source_case in (source / "cases").glob("*.npz"):
    os.link(source_case, target / "cases" / source_case.name)
shutil.copy2(source / "metadata.json", target / "metadata.json")
PY
  fi
fi
checkpoint="$output_dir/checkpoints/latest.pt"
resume_args=()
if [[ -f "$checkpoint" ]]; then
  resume_args=(--resume "$checkpoint")
fi

"$cue_python" scripts/run_cue_sfda.py train \
  --config "$config" \
  --output-dir "$output_dir" \
  --consensus-mode "$mode" \
  --consensus-cache "$consensus_cache" \
  --max-iterations 2000 \
  --gpu-id "$gpu" \
  --gt-path "$gt_path" \
  "${resume_args[@]}"

summary_args=(--output-dir "$root/summary" --baseline "$root/A_C_baseline")
for stage in \
  E_clean_language \
  F_dynamic_hard_language \
  G_selective_dynamic_refresh \
  H_selective_dynamic_language; do
  stage_state="$root/$stage/early_stopping.json"
  if [[ -f "$stage_state" ]] && ! rg -q '"stop_iteration": null' "$stage_state"; then
    summary_args+=(--experiment "$stage=$root/$stage")
  fi
done
"$cue_python" scripts/summarize_ablation.py "${summary_args[@]}"
