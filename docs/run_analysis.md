# Run analysis

## DWI-only reproduction status

All values are mean Dice on the configured 44-case DWI test split. Training
losses use only the 101 target images. The prompt strings remain exactly
`liver` and `liver cancer`.

| task | inference path | mean Dice | output |
|---|---|---:|---|
| liver | frozen DWI VoxTell | 0.732752 | `outputs/dwi_only_liver_expanded/baseline_metrics` |
| liver | DWI-only CUE + expanded LoRA, iteration 100 | **0.732843** | `outputs/dwi_only_liver_expanded/evaluation` |
| liver cancer | frozen DWI VoxTell | 0.283336 | `outputs/dwi_only_liver_cancer_expanded/baseline_metrics` |
| liver cancer | DWI-only CUE + expanded LoRA, iteration 400 | **0.311355** | `outputs/dwi_only_liver_cancer_expanded/evaluation` |

Earlier reported values 0.758672 and 0.378390 used a paired PRE sequence and
are invalid under the DWI-only constraint. That evaluation branch and its
outputs have been removed. New DWI-only experiments are written below
`outputs/dwi_only_liver*`.

The requested 0.75/0.37 thresholds are not reached under this constraint. Even
an evaluation-only oracle that selects the better source/adapted mask for each
case reaches only 0.737268 for liver and 0.336654 for liver cancer. This oracle
uses labels only to establish an upper bound and is not part of the method.

## What was wrong before

1. Cases were divided by a median quantile. The paper instead computes Dice
   between the raw and uncertainty-filtered pseudo-label and compares it to a
   fixed modality threshold gamma (0.5 for MRI).
2. The former uncertainty threshold was 0.05. The paper's selected hesitation
   threshold is 0.8, and it uses K=10 strongly augmented, pairwise-mixed views.
3. The earlier "instance style normalization" was another input z-score. The
   paper actually learns an FFT low-frequency attention map using source BN
   running statistics. VoxTell uses InstanceNorm3d and has no such statistics,
   so that objective is undefined and the incorrect substitute is disabled.
4. The previous random walk used six neighbours, cosine affinity, restart, and
   five steps. The paper uses feature and image affinity, an unanchored
   transition, and four steps. The reproduction now uses the corresponding
   3-D 26-neighbour graph.
5. Low-quality consistency used two-view MSE. The paper uses KL divergence from
   the multi-view mean prototype. The implementation now uses Bernoulli KL.
6. Cancer had many empty or near-empty DWI predictions. Self-training can
   refine existing masks but has no positive seed in these cases. This is the
   practical reason the paper-only single-DWI runs remained below the requested
   cancer target.

## Deviations from the paper

- CUE-SFDA was specified for 2-D DeepLabv3+/MobileNetV2. VoxTell is a 440M
  parameter, text-prompted 3-D network. The local affinity graph is therefore
  downsampled and extended from a 2-D radius neighbourhood to 3-D 26-neighbour
  propagation.
- The paper's FFT/BN low-frequency style-transfer objective cannot be reproduced
  on a checkpoint whose normalizers are InstanceNorm3d. It is disabled rather
  than replaced with a mathematically different operation.
- Rank-8 adapters cover Q/K/V plus text and final-mask projections to fit a
  192-cubed patch on an 11 GB GPU. The low-quality branch forms its detached
  prototype from three views and retains one differentiable view; retaining ten
  3-D VoxTell graphs simultaneously exceeds available memory.
- The 3-D graph preserves raw foreground seeds after pooled affinity propagation
  so small tumors are not erased by graph downsampling. Reliable cases are
  sampled more frequently for the sparse tumor task. Both are label-free 3-D
  engineering extensions rather than claims made by the 2-D paper.
- The reproduction consumes only DWI images. The removed PRE fusion was not a
  component of the paper and is no longer present in code or configuration.

Labels are read only by `evaluate`/periodic validation after predictions have
been produced. They never enter uncertainty calibration, pseudo-label creation,
losses, sampling, or parameter updates.
