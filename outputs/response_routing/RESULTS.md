# Response-aware multi-source uncertainty routing

All reported adapted results use the iteration-1000 `latest.pt` checkpoint,
not a test-set-selected best checkpoint. Frozen VoxTell baseline Dice is
`0.283335689`, matching the existing zero-shot result (`0.283336`).

| Experiment | Routing | Dice | mIoU | Precision | Recall | Dice vs A |
|---|---|---:|---:|---:|---:|---:|
| A | Original CUE-SFDA | 0.323586 | 0.246036 | 0.321822 | 0.475190 | 0.000000 |
| B | Image uncertainty | 0.321493 | 0.244903 | 0.303828 | 0.504495 | -0.002093 |
| C | Image + Prompt, no gate | 0.319205 | 0.240669 | 0.272248 | 0.582921 | -0.004381 |
| D | Image + Prompt + No-response gate | **0.328358** | **0.248156** | 0.283514 | 0.571524 | **+0.004772** |

## Baseline-Dice subgroups

| Experiment | Baseline group | n | Baseline Dice | Adapted Dice | Delta |
|---|---:|---:|---:|---:|---:|
| A | <0.1 | 22 | 0.001815 | 0.075754 | +0.073939 |
| A | 0.1-0.5 | 10 | 0.330789 | 0.338616 | +0.007827 |
| A | >0.5 | 12 | 0.759912 | 0.765419 | +0.005507 |
| B | <0.1 | 22 | 0.001815 | 0.089380 | +0.087565 |
| B | 0.1-0.5 | 10 | 0.330789 | 0.349953 | +0.019164 |
| B | >0.5 | 12 | 0.759912 | 0.723317 | -0.036594 |
| C | <0.1 | 22 | 0.001815 | 0.117010 | +0.115194 |
| C | 0.1-0.5 | 10 | 0.330789 | 0.324380 | -0.006409 |
| C | >0.5 | 12 | 0.759912 | 0.685584 | -0.074328 |
| D | <0.1 | 22 | 0.001815 | **0.114377** | **+0.112561** |
| D | 0.1-0.5 | 10 | 0.330789 | 0.320820 | -0.009969 |
| D | >0.5 | 12 | 0.759912 | 0.726938 | -0.032974 |

Routing counts are A/B/C: high=59, low=42, skip=0; D: stable=49,
unstable=35, no-response/skip=17. A separate frozen full-volume inference
confirmed that all 17 no-response cases have exactly zero predicted foreground
voxels. No empty test GT, NaN loss, all-zero training loss, or all-empty model
collapse was detected. HD95/ASD are not implemented by the existing evaluator.

## Commands

```bash
export PYTHONPATH=src
PY=/home/ubuntu/anaconda3/envs/voxtell/bin/python
CFG=configs/response_routing_liver_cancer.yaml
GT=/data/hilab/datasets/WCH_MRI145/DWI/labels

# A (the completed original run is retained under outputs/dwi_only_liver_cancer)
$PY -m cue_sfda.cli train --config configs/dwi_liver_cancer.yaml \
  --output-dir outputs/dwi_only_liver_cancer --gpu-id 0 --gt-path $GT
$PY -m cue_sfda.cli train --config $CFG --routing-mode image \
  --output-dir outputs/response_routing/B_image --gpu-id 0 --gt-path $GT
$PY -m cue_sfda.cli train --config $CFG --routing-mode joint \
  --output-dir outputs/response_routing/C_joint --gpu-id 1 --gt-path $GT
$PY -m cue_sfda.cli train --config $CFG --routing-mode joint_gate \
  --output-dir outputs/response_routing/D_joint_gate --gpu-id 0 --gt-path $GT

$PY -m cue_sfda.cli evaluate --config $CFG \
  --output-dir outputs/response_routing/baseline --gpu-id 1 --gt-path $GT
$PY scripts/check_no_response.py --config $CFG \
  --routing-csv outputs/response_routing/D_joint_gate/routing.csv \
  --output-dir outputs/response_routing/no_response_check --gpu-id 1

$PY scripts/summarize_response_routing.py \
  --baseline-dir outputs/response_routing/baseline \
  --experiment A outputs/dwi_only_liver_cancer \
  --experiment B outputs/response_routing/B_image \
  --experiment C outputs/response_routing/C_joint \
  --experiment D outputs/response_routing/D_joint_gate \
  --output-dir outputs/response_routing/summary
```
