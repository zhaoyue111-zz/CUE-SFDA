# Stage 2 voxel-level uncertainty analysis

## Scope and protocol

All scores were computed offline from the 101 saved multiview probability caches; there was no model inference, training, parameter update, adaptation, or gradient discount. GT is used only to select/visualize cases and evaluate maps. Per-case macro statistics are reported so larger volumes do not dominate. The upstream image-level historical baseline audit remains unresolved (mean Dice difference about 6e-4 after checking the repository's original VoxTellPredictor); this voxel analysis is conditional on the current benchmark outputs.

The 24 cases are fixed in `selected_cases.csv`: 8 historical E-High and currently nonempty Dice-zero cases; 4 currently empty Dice-zero cases; 4 in Dice [0.1,0.5]; 4 in (0.5,0.8]; and 4 above 0.8. Case status uses the current benchmark. The selected cases have 1 historical/current prediction-empty status changes and 0 Dice-bin changes; both are recorded per case. All cases use the same selected slice across every map, listed in `visualization_slices.csv`.

P0 and P3–P8 were computed for Image-only and Prompt-only independently. Image+Prompt is the equal-weight voxel-map mean `0.5*Image + 0.5*Prompt`, preserving the requested modality balance. P1 MPC and P2 entropy use the original standard-prompt prediction and are marked `Original`; P9 is Image–Prompt mean-probability disagreement. Formulas follow `src/cue_sfda/benchmark/metrics.py`; no spatial aggregation is used. P7 is reported in nats. AUROC/AUPRC use a 16,384-bin fixed-range histogram approximation for scalable per-case evaluation. The fixed mathematical display ranges used by visualizations are encoded in the analysis script; no slice-wise min–max normalization is applied.

The three evaluation regions are full valid inference ROI, original prediction foreground, and the union of original/image-view/prompt-view foreground. The binary error target is FP or FN. `per_case_metrics.csv` records case-level AUROC/AUPRC and FP/FN coverage at top 1%, 5%, 10%, and 20% uncertainty. When an evaluation region has no positives or negatives, the corresponding undefined score is NaN with a reason. `region_uncertainty_statistics.csv` gives TP/TN/FP/FN uncertainty distributions for each case, map, and region. `special_case_response.csv` separately summarizes the 22 zero-Dice nonempty predictions and empty-prediction cases with GT lesions, including lesion FN uncertainty in the full ROI and multiview union.

## Case-macro full valid ROI summary

| Metric | AUROC | AUPRC | FP mean U | FN mean U | TP mean U | TN mean U |
|---|---:|---:|---:|---:|---:|---:|
| P0_Chebyshev | 0.7731 | 0.3008 | 0.4703 | 0.1488 | 0.2735 | 0.0004834 |
| P1_MPC | 0.6667 | 0.1551 | 0.05717 | 0.02289 | 0.03315 | 4.047e-05 |
| P2_Entropy | 0.6725 | 0.1591 | 0.1914 | 0.07717 | 0.1074 | 0.0001453 |
| P3_Predictive_Entropy | 0.8063 | 0.3239 | 0.5479 | 0.1923 | 0.3247 | 0.0007052 |
| P4_Expected_Entropy | 0.7943 | 0.2470 | 0.1339 | 0.05986 | 0.06672 | 0.0001975 |
| P5_Predictive_Variance | 0.7730 | 0.2898 | 0.09955 | 0.03044 | 0.06054 | 0.000106 |
| P6_Mutual_Information | 0.7975 | 0.3004 | 0.414 | 0.1325 | 0.258 | 0.0005076 |
| P7_EPKL | 0.7856 | 0.2670 | 1.88 | 0.5392 | 1.285 | 0.002071 |
| P8_Vote_Disagreement | 0.7422 | 0.2734 | 0.1932 | 0.05713 | 0.1059 | 0.0001764 |
| P9_Image_Prompt_Disagreement | 0.7958 | 0.2800 | 0.3159 | 0.1051 | 0.2698 | 0.0003652 |

Read modality-specific comparisons, prediction-foreground/union analyses, top-uncertainty coverage, and per-case values from `metric_summary.csv`, `per_case_metrics.csv`, and `error_coverage_summary.csv`. These are descriptive; there is no automatically selected winner.

## Visualization and spatial files

Each selected case has a fixed-range overlay montage at `visualizations/<case>.png` with MRI, GT/Prediction contours, TP/FP/FN colors, and all applicable P0–P9 uncertainty overlays, including modality comparisons. MRI display window is common across selected cases (pooled ROI 1st–99th percentile: 1 to 212); overlay alpha and magma color map are common. Each metric uses its fixed theoretical display range, never per-slice normalization.

`uncertainty_maps/<case>/` contains each selected case's complete 3D `.nii.gz` maps plus MRI reference, GT, prediction, and valid ROI layers. Files are written in the MRI reader's canonical reoriented geometry with the same physical affine; non-ROI voxels are zero. Load `MRI_reference.nii.gz` as the base in ITK-SNAP and add the maps/labels as overlays.

## Reading the results

The summaries compare error responses (FP/FN) with correct regions (TP/TN) for each perturbation, and show whether uncertainty concentrates on errors in the full ROI, foreground, and multiview union. Empty-prediction cases have no prediction-foreground region, so those region metrics are undefined; their GT lesion response is evaluated in the full ROI and multiview union. No single metric is declared best. Choose based on these macro statistics and the aligned visualizations.
