# Image/Prompt uncertainty fusion weight ablation

## Scope and method

This is an offline exploratory sweep using only the existing `per_case_scores.csv`; there was no inference, model training, parameter update, or adaptation. The 101 cases and their GT Dice values come from the current benchmark. Therefore, alpha selection and all comparisons are exploratory and use training-set GT; they are not an unbiased held-out evaluation. This sweep does not resolve the upstream historical baseline discrepancy: the prior audit records a mean absolute Dice difference of about `6e-4`, which remained unexplained after checking with the repository's original `VoxTellPredictor`. Treat these results as conditional comparisons within the current score table, not as a validated reproduction of the historical experiment.

Low-Dice detection uses uncertainty (`−confidence`) as the positive-class score; Dice>0.5 detection uses confidence. Gated empty predictions receive confidence 0 and sort after every nonempty case even if nonempty scores are tied or nonpositive. Nonempty-case ranks, Top-K means, High/Low groups, and zero-Dice error counts are recalculated within the 81-case nonempty subset. Original CUE-DIC now has its own all-case/nonempty ranks and Top-59 flags.

## Provisional candidates for later validation

| Role | Candidate |
|---|---|
| Main | Pairwise Hard Dice, Image 0.4 / Prompt 0.6 (α=0.6) |
| Lightweight | Pairwise Hard Dice, Prompt-only (α=1.0) |
| Control | Pairwise Soft Dice, Image 0.5 / Prompt 0.5 |
| Baseline | Original E |

For each metric, fusion was performed on the uncertainty values, `U_joint=(1−α)U_image+αU_prompt`, for α=0.0…1.0 by 0.1. Confidence was then recomputed using the existing metric transform: `1−U_joint` for I1/I2 and `1/(1+U_joint)` for I3 VVC. Gated results set confidence to 0 for empty predictions; raw results retain the raw confidence. There are 101 total cases, 81 nonempty predictions, 20 empty predictions, and 22 zero-Dice nonempty predictions. High is the top 59 by confidence, Low the remaining 42; ties are broken by case ID. Dice bands are `<0.1`, `[0.1,0.5]`, and `>0.5`.

## Consistency checks and controls

At α=0, 0.5, and 1.0, scores reproduce the existing Image-only, Image+Prompt, and Prompt-only confidence transforms to numerical precision (see `alpha_anchor_validation.csv`). α=0.5 is the existing E-style equal fusion control; α=1.0 is pure text. Original CUE-DIC is retained separately as a routing control in `per_case_scores.csv` and `fusion_summary.csv`. Gated empty predictions use the requested confidence 0 convention.

## Best AURC candidate by metric and gating

| Metric | Gating | α | AURC | Spearman (101) | Top-59 mean Dice |
|---|---|---:|---:|---:|---:|
| I1_Pairwise_Hard | raw | 0.4 | 0.5277 | 0.5592 | 0.5080 |
| I1_Pairwise_Hard | gated | 1.0 | 0.3825 | 0.8630 | 0.5615 |
| I2_Pairwise_Soft | raw | 0.5 | 0.4949 | 0.6609 | 0.5218 |
| I2_Pairwise_Soft | gated | 1.0 | 0.3863 | 0.8550 | 0.5615 |
| I3_VVC | raw | 0.7 | 0.5127 | 0.6333 | 0.5227 |
| I3_VVC | gated | 0.9 | 0.3906 | 0.8514 | 0.5615 |

Full and nonempty Spearman, Pearson, AURC, AUROC/AUPRC, all requested Top-K means, and independently calculated High/Low Dice are in `fusion_summary.csv`. The three Dice-band counts and within-group proportions for all-case and nonempty High/Low sets are in `high_low_dice_bins.csv`. Confidence, all/nonempty cohort ranks, both Top-59 flags, and the 22 zero-Dice nonempty cases are in `per_case_scores.csv` and `zero_dice_nonempty_cases.csv`. Original CUE-DIC all-case/nonempty ranks and Top-59 membership are included in the same per-case table.

## Paired bootstrap comparisons

For each metric and gating mode, the candidate α minimizing observed all-case AURC was compared with α=0.5 and α=1.0 using 2,000 paired case bootstrap resamples. The intervals quantify uncertainty conditional on this cohort and the selected candidate; because candidate selection and assessment use the same GT cohort, they do not establish generalization.

| Metric | Gating | Candidate α | Comparator α | Statistic (candidate−comparator) | 95% paired CI | Stable improvement |
|---|---|---:|---:|---|---:|---|
| I1_Pairwise_Hard | raw | 0.4 | 0.5 | spearman | 0.0037 [-0.0182, 0.0258] | 否 |
| I1_Pairwise_Hard | raw | 0.4 | 0.5 | pearson | -0.0109 [-0.0280, 0.0058] | 否 |
| I1_Pairwise_Hard | raw | 0.4 | 0.5 | aurc | -0.0030 [-0.0116, 0.0029] | 否 |
| I1_Pairwise_Hard | raw | 0.4 | 0.5 | top59_mean_dice | -0.0164 [-0.0310, 0.0132] | 否 |
| I1_Pairwise_Hard | raw | 0.4 | 1.0 | spearman | 0.2069 [0.0574, 0.3612] | 是 |
| I1_Pairwise_Hard | raw | 0.4 | 1.0 | pearson | 0.0503 [-0.0654, 0.1619] | 否 |
| I1_Pairwise_Hard | raw | 0.4 | 1.0 | aurc | -0.1078 [-0.1758, -0.0459] | 是 |
| I1_Pairwise_Hard | raw | 0.4 | 1.0 | top59_mean_dice | 0.0020 [-0.0378, 0.0596] | 否 |
| I1_Pairwise_Hard | gated | 1.0 | 0.5 | spearman | 0.0113 [-0.0231, 0.0499] | 否 |
| I1_Pairwise_Hard | gated | 1.0 | 0.5 | pearson | -0.0408 [-0.0784, -0.0033] | 否 |
| I1_Pairwise_Hard | gated | 1.0 | 0.5 | aurc | -0.0108 [-0.0257, 0.0003] | 否 |
| I1_Pairwise_Hard | gated | 1.0 | 0.5 | top59_mean_dice | -0.0008 [-0.0165, 0.0241] | 否 |
| I1_Pairwise_Hard | gated | 1.0 | 1.0 | spearman | 0.0000 [0.0000, 0.0000] | 否 |
| I1_Pairwise_Hard | gated | 1.0 | 1.0 | pearson | 0.0000 [0.0000, 0.0000] | 否 |
| I1_Pairwise_Hard | gated | 1.0 | 1.0 | aurc | 0.0000 [0.0000, 0.0000] | 否 |
| I1_Pairwise_Hard | gated | 1.0 | 1.0 | top59_mean_dice | 0.0000 [0.0000, 0.0000] | 否 |
| I2_Pairwise_Soft | raw | 0.5 | 0.5 | spearman | 0.0000 [0.0000, 0.0000] | 否 |
| I2_Pairwise_Soft | raw | 0.5 | 0.5 | pearson | 0.0000 [0.0000, 0.0000] | 否 |
| I2_Pairwise_Soft | raw | 0.5 | 0.5 | aurc | 0.0000 [0.0000, 0.0000] | 否 |
| I2_Pairwise_Soft | raw | 0.5 | 0.5 | top59_mean_dice | 0.0000 [0.0000, 0.0000] | 否 |
| I2_Pairwise_Soft | raw | 0.5 | 1.0 | spearman | 0.2426 [0.0974, 0.3907] | 是 |
| I2_Pairwise_Soft | raw | 0.5 | 1.0 | pearson | 0.1192 [0.0371, 0.2052] | 是 |
| I2_Pairwise_Soft | raw | 0.5 | 1.0 | aurc | -0.1204 [-0.1925, -0.0566] | 是 |
| I2_Pairwise_Soft | raw | 0.5 | 1.0 | top59_mean_dice | 0.0090 [-0.0209, 0.0568] | 否 |
| I2_Pairwise_Soft | gated | 1.0 | 0.5 | spearman | -0.0025 [-0.0435, 0.0356] | 否 |
| I2_Pairwise_Soft | gated | 1.0 | 0.5 | pearson | -0.0640 [-0.1053, -0.0233] | 否 |
| I2_Pairwise_Soft | gated | 1.0 | 0.5 | aurc | -0.0066 [-0.0226, 0.0075] | 否 |
| I2_Pairwise_Soft | gated | 1.0 | 0.5 | top59_mean_dice | -0.0056 [-0.0262, 0.0155] | 否 |
| I2_Pairwise_Soft | gated | 1.0 | 1.0 | spearman | 0.0000 [0.0000, 0.0000] | 否 |
| I2_Pairwise_Soft | gated | 1.0 | 1.0 | pearson | 0.0000 [0.0000, 0.0000] | 否 |
| I2_Pairwise_Soft | gated | 1.0 | 1.0 | aurc | 0.0000 [0.0000, 0.0000] | 否 |
| I2_Pairwise_Soft | gated | 1.0 | 1.0 | top59_mean_dice | 0.0000 [0.0000, 0.0000] | 否 |
| I3_VVC | raw | 0.7 | 0.5 | spearman | 0.0001 [-0.0302, 0.0273] | 否 |
| I3_VVC | raw | 0.7 | 0.5 | pearson | 0.0047 [-0.0228, 0.0332] | 否 |
| I3_VVC | raw | 0.7 | 0.5 | aurc | -0.0028 [-0.0093, 0.0027] | 否 |
| I3_VVC | raw | 0.7 | 0.5 | top59_mean_dice | -0.0168 [-0.0374, 0.0276] | 否 |
| I3_VVC | raw | 0.7 | 1.0 | spearman | 0.3017 [0.1302, 0.4752] | 是 |
| I3_VVC | raw | 0.7 | 1.0 | pearson | 0.1386 [0.0462, 0.2413] | 是 |
| I3_VVC | raw | 0.7 | 1.0 | aurc | -0.1290 [-0.2048, -0.0639] | 是 |
| I3_VVC | raw | 0.7 | 1.0 | top59_mean_dice | 0.0229 [-0.0122, 0.0736] | 否 |
| I3_VVC | gated | 0.9 | 0.5 | spearman | 0.0273 [-0.0116, 0.0750] | 否 |
| I3_VVC | gated | 0.9 | 0.5 | pearson | -0.0195 [-0.0490, 0.0110] | 否 |
| I3_VVC | gated | 0.9 | 0.5 | aurc | -0.0123 [-0.0260, -0.0006] | 是 |
| I3_VVC | gated | 0.9 | 0.5 | top59_mean_dice | 0.0051 [-0.0226, 0.0298] | 否 |
| I3_VVC | gated | 0.9 | 1.0 | spearman | 0.0126 [-0.0142, 0.0398] | 否 |
| I3_VVC | gated | 0.9 | 1.0 | pearson | 0.0139 [0.0034, 0.0259] | 是 |
| I3_VVC | gated | 0.9 | 1.0 | aurc | -0.0026 [-0.0139, 0.0108] | 否 |
| I3_VVC | gated | 0.9 | 1.0 | top59_mean_dice | 0.0000 [-0.0105, 0.0320] | 否 |

## Interpretation

The raw AURC-selected interior mixtures improve point estimates over both single-source endpoints for all three metrics: I1_Pairwise_Hard: α=0.4 AURC 0.5277, vs image-only α=0 -0.0214, vs pure text α=1 -0.1078; I2_Pairwise_Soft: α=0.5 AURC 0.4949, vs image-only α=0 -0.0139, vs pure text α=1 -0.1204; I3_VVC: α=0.7 AURC 0.5127, vs image-only α=0 -0.0132, vs pure text α=1 -0.1290. This supports image/text complementarity in this cohort. Against pure text α=1, raw paired-bootstrap AURC and Spearman improvements are stable for the selected candidates, while Top-59 Dice gains are not stable. Against equal fusion α=0.5, raw AURC differences are not stable for I1 and I3; I2 selects α=0.5 itself. In gated results the AURC-minimizing weights are I1_Pairwise_Hard α=1.0, I2_Pairwise_Soft α=1.0, I3_VVC α=0.9; only VVC at α=0.9 has a stable AURC improvement over gated α=0.5 in this sweep. Thus increasing text weight above 0.5 does not give a consistent improvement across metrics and gating modes.

The most defensible weight to retain is α=0.5: it is the raw AURC optimum for I2, lies close to the raw optima for I1 and I3 without statistically stable gains over 0.5, and is the existing equal-fusion E control. Higher text weights remain candidates for a held-out check, particularly with gating, but are not established as a universal replacement. All these conclusions are exploratory because alpha was selected and evaluated against the same training-set GT cohort.

## Figures

`figures/spearman_vs_alpha.png`, `figures/aurc_vs_alpha.png`, and `figures/topk_mean_dice_vs_alpha.png` show the alpha curves. `figures/confidence_dice_scatter_raw.png` and `figures/confidence_dice_scatter_gated.png` show Confidence–Dice distributions for all fusion weights.
