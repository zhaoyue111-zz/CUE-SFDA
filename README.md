# CUE-SFDA with VoxTell

This repository reproduces **Consistency-Guided Uncertainty Estimation for
Source-Free Medical Image Segmentation** with the pretrained VoxTell model. The
VoxTell source tree is imported as an external dependency and is never modified.

The implementation follows the paper's two-stage method:

1. aligned predictions from strongly augmented and style-mixed target views
   produce a Chebyshev error bound;
2. target cases are automatically split into high- and low-confidence subsets;
3. high-confidence patches use image/feature affinity and four-step random walk
   refinement;
4. low-confidence patches use KL consistency with a multi-view prototype.

Only LoRA matrices attached to Q, K and V in VoxTell's prompt transformer
decoder are trainable. Ground truth never enters an adaptation loss. The test
cases are evaluated every 20 iterations and once after the final iteration.

## Layout

```text
configs/                 experiment configuration
scripts/                 command-line entry point
src/cue_sfda/data/       NIfTI loading and sliding-window patch generation
src/cue_sfda/modeling/   VoxTell adapter and decoder-QKV LoRA
src/cue_sfda/method/     uncertainty, augmentation and graph refinement
src/cue_sfda/engine/     1000-iteration adaptation loop
src/cue_sfda/evaluation.py
tests/                   lightweight unit tests
```

The method-to-code mapping and reproduction assumptions are documented in
[`docs/method.md`](docs/method.md). Verified Dice results, formula-level
corrections, and deviations are in
[`docs/run_analysis.md`](docs/run_analysis.md).

## Run

The default configuration adapts on the 101 DWI training cases for exactly
1000 iterations and evaluates on the 44 test cases. It also evaluates the
unadapted model at iteration 0, so adaptation results can always be compared
against the actual starting point.

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH}"

python scripts/run_cue_sfda.py train \
  --config configs/dwi_liver_cancer.yaml \
  --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels
```

Select the CUDA card with `experiment.gpu_id` in the YAML file or override it
with `--gpu-id`. Periodic validation writes NIfTI masks below its iteration
directory. The checkpoint directory contains only `latest.pt` and one best
checkpoint named with its iteration and validation Dice.

For liver segmentation, use its separate config (the prompt remains `liver`):

```bash
python scripts/run_cue_sfda.py train \
  --config configs/dwi_liver.yaml \
  --gt-path /data/hilab/datasets/WCH_MRI145/DWI/liver \
  --output-dir outputs/dwi_only_liver_expanded
```

Evaluation can also be run separately:

```bash
python scripts/run_cue_sfda.py evaluate \
  --config configs/dwi_liver_cancer.yaml \
  --checkpoint outputs/dwi_only_liver_cancer_expanded/checkpoints/best_iteration_000400_dice_0.311355.pt \
  --output-dir outputs/dwi_only_liver_cancer_eval \
  --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels
```

Useful smoke-test overrides are `--iterations 2`, `--calibration-cases 2`, and
`--skip-eval`. Prompt embeddings are cached below the output directory, so the
4B text encoder is only loaded on a cache miss.

## Reproduction note

The local PDF supplies the complete equations even though the official
repository has no implementation. The Chebyshev map is

```text
B(x) = sigma(x)^2 / (sigma(x)^2 + (mu(x) - tau)^2),
```

where `mu` and `sigma` come from K=10 mixed strong views and `tau` in this
display is the segmentation threshold (`delta` in the paper). Voxels below the
separate hesitation threshold 0.8 are reliable. Case quality is the Dice between
the raw and filtered pseudo-label, compared with fixed MRI gamma=0.5.

The supplied VoxTell checkpoint uses InstanceNorm3d, so the paper's BN-statistic
FFT style objective is undefined and is not replaced by a misleading input
normalization. Only the configured DWI images are used. No auxiliary sequence,
external dataset, or target annotation is available to adaptation. Target
labels are read only by the explicitly requested evaluation step.

## High 分支更新接纳对照

`stage1-quality-routing` 分支提供四个独立 M1 Dynamic-hard High 更新接纳配置（GA、CUV、CSS、SUV）。它们复用原固定 High/Low 路由和 Low Dynamic-hard 训练流程，只决定当前 High 更新是否进入共享 Student。理论假设、缓存、计算量、参数和启动方式见 [`docs/high_selection.md`](docs/high_selection.md)。
