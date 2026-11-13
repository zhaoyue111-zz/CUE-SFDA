export PYTHONPATH=src
PY=/home/ubuntu/anaconda3/envs/voxtell/bin/python
GT=/data/hilab/datasets/WCH_MRI145/DWI/labels
CFG=configs/response_routing_liver_cancer.yaml

$PY -m cue_sfda.cli evaluate \
  --config configs/dwi_liver_cancer.yaml \
  --checkpoint outputs/dwi_only_liver_cancer/checkpoints/best_iteration_000700_dice_0.330541.pt \
  --output-dir outputs/dwi_only_liver_cancer/best_evaluation \
  --gpu-id 0 \
  --gt-path "$GT"

$PY -m cue_sfda.cli evaluate \
  --config "$CFG" \
  --checkpoint outputs/response_routing/B_image/checkpoints/best_iteration_000900_dice_0.330546.pt \
  --output-dir outputs/response_routing/B_image/best_evaluation \
  --gpu-id 0 \
  --gt-path "$GT"

$PY -m cue_sfda.cli evaluate \
  --config "$CFG" \
  --checkpoint outputs/response_routing/C_joint/checkpoints/best_iteration_000800_dice_0.338383.pt \
  --output-dir outputs/response_routing/C_joint/best_evaluation \
  --gpu-id 1 \
  --gt-path "$GT"

$PY -m cue_sfda.cli evaluate \
  --config "$CFG" \
  --checkpoint outputs/response_routing/D_joint_gate/checkpoints/best_iteration_000800_dice_0.329318.pt \
  --output-dir outputs/response_routing/D_joint_gate/best_evaluation \
  --gpu-id 1 \
  --gt-path "$GT"

$PY scripts/summarize_response_routing.py \
  --baseline-dir outputs/response_routing/baseline \
  --experiment A outputs/dwi_only_liver_cancer \
  --experiment B outputs/response_routing/B_image \
  --experiment C outputs/response_routing/C_joint \
  --experiment D outputs/response_routing/D_joint_gate \
  --output-dir outputs/response_routing/best_summary