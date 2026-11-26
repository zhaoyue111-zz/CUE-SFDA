# Stage 1 Image-level Uncertainty Benchmark

## Baseline audit

All 101 case IDs match the historical file. Mean absolute Dice difference is 0.001176; maximum is 0.023718 for case `0019971803` (historical 0.125000, current 0.148718). The 1e-4 audit threshold is exceeded, as accepted for this run; findings are reported without treating the runs as numerically identical.
Current prediction counts: 20 empty, 81 nonempty, 22 Dice-zero/nonempty. Expected historical counts were 17, 84, and 25. Dice bins <0.1 / [0.1,0.5] / >0.5 are 49 / 12 / 40.

The largest case-level deviations and empty/nonempty transitions are in `baseline_reproducibility.csv`; review these before interpreting fine-grained deltas.

## Leading candidates

Ranking considers nonempty-case Spearman and full-case AURC; it is descriptive, not a trained calibration.

| Metric | Views | Aggregate | Spearman (nonempty) | AURC (all) | Dice-zero/nonempty in Top-59 |
|---|---|---:|---:|---:|---:|
| I1_Pairwise_Hard | Prompt-only | case | 0.836 | 0.635 | 3 |
| I2_Pairwise_Soft | Prompt-only | case | 0.821 | 0.615 | 4 |
| I2_Pairwise_Soft | Image+Prompt | case | 0.817 | 0.495 | 3 |
| I1_Pairwise_Hard | Image+Prompt | case | 0.811 | 0.531 | 3 |
| I3_VVC | Prompt-only | case | 0.791 | 0.642 | 4 |

## Metric recommendations

These are candidates for later gradient-discount validation, not calibrated absolute quality probabilities. The recommendation weighs nonempty-case association, full-cohort risk-coverage, failure detection, Top-59 mistakes, ties, and computation cost.

| Candidate | Nonempty Spearman | AURC | AUROC / AUPRC for Dice < 0.1 | Raw Top-59 mean Dice / zero-nonempty | Gated Top-59 mean Dice / zero-nonempty |
|---|---:|---:|---:|---:|---:|
| I2_Pairwise_Soft / Image+Prompt | 0.817 | 0.495 | 0.841 / 0.876 | 0.522 / 3 | 0.567 / 7 |
| I1_Pairwise_Hard / Prompt-only | 0.836 | 0.635 | 0.655 / 0.762 | 0.506 / 3 | 0.562 / 9 |
| I3_VVC / Image+Prompt | 0.765 | 0.515 | 0.829 / 0.879 | 0.540 / 4 | 0.556 / 7 |
| I2_Pairwise_Soft / Image-only | 0.712 | 0.509 | 0.818 / 0.839 | 0.500 / 7 | 0.521 / 8 |

I2 soft Dice with Image+Prompt has the strongest AURC and failure detection among these choices and keeps the continuous score nearly tie-free. I1 hard Dice with Prompt-only has the highest nonempty Spearman but has more ties and weaker failure AUROC. I3 VVC with Image+Prompt and I2 soft Dice with Image-only are useful alternatives when view-group robustness or image-only operation matters.

For the eight historically misrouted E cases, gated Top-59 retention is: I2_Pairwise_Soft/Image+Prompt: 7/8 of the previously misrouted E cases remain in gated Top-59; I1_Pairwise_Hard/Prompt-only: 8/8 of the previously misrouted E cases remain in gated Top-59; I3_VVC/Image+Prompt: 6/8 of the previously misrouted E cases remain in gated Top-59; I2_Pairwise_Soft/Image-only: 5/8 of the previously misrouted E cases remain in gated Top-59.

## Perturbation and spatial aggregation

Image+Prompt means the two group scores were computed independently and averaged. It is not a pooled-view score. For voxel-map methods, Union Mean often improves nonempty ranking over Full-volume Mean: for example, prompt-only PE Spearman changes from -0.025 (G0) to 0.376 (G1), and prompt-only EPKL from 0.346 to 0.517. This gain does not generalize to every map; review `conditional_correlations.csv` and `aurc_summary.csv` together.

## A/E reproduction

A: 59 High / 42 Low; mean Dice 0.530324 / 0.066640.
E: 59 High / 42 Low; mean Dice 0.564629 / 0.018449. E empty-to-low summary: 17 empty and 84 nonempty. E Top-59 case IDs match historical `routing.csv`: True.

## Stable errors, confidence, and lesion size

Current predictions include 22 Dice-zero/nonempty cases (historical count: 25). The case-level table lists each metric's uncertainty, confidence, cohort rank, nonempty rank, Top-59 status, and A/E rank for these cases and the eight previously selected E errors.

Confidence quintiles and distributions are in `score_bin_summary.csv` and `figures/`. GT/prediction physical volumes and view-wise foreground volumes are in `lesion_size_analysis.csv`; volume correlations and tertile results are in `lesion_size_correlations.csv` and `lesion_size_group_summary.csv`. GT volumes are used only in this offline analysis.

## Outputs

Per-case scores retain raw and gated results. Detailed correlations, conditional subsets, AURC, raw and gated Top-K, High/Low, paired bootstrap, failure cases, score bins, lesion volumes, and figures are stored alongside this report.

## Caveats

The baseline is not numerically aligned to 1e-4; mean/max absolute Dice differences are reported above. The E historical empty count is 17 versus 20 in this run; additional current-empty cases are 0009011272, 0018689853, 0019218185. The user authorized continuing; interpret comparisons with this discrepancy visible. Four cases have empty G1 unions; they are recorded with NaN uncertainty and zero valid voxels in `failure_log.csv`, with no imputation. No GT was used in uncertainty computation; GT enters only post-inference evaluation and analysis. No adaptation or training was performed.
