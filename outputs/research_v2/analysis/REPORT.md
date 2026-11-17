# Research v2 Stage A/B results

The 44-case held-out split was reused for earlier method and checkpoint selection. These fixed-iteration scores are exploratory labeled development results, not an untouched final test. No checkpoint was selected using these metrics.

## Aggregate metrics

| Run | Dice | Precision | Recall | IoU |
|---|---:|---:|---:|---:|
| E0 | 0.283428 | 0.369238 | 0.336402 | 0.217554 |
| E1 | 0.303781 | 0.345651 | 0.384488 | 0.230605 |
| H1v1 | 0.303190 | 0.346000 | 0.382971 | 0.230190 |

## Paired Dice changes

| Contrast | Mean Δ | Median Δ | Bootstrap 95% CI | Improved | Unchanged | Declined | Δ>0.02 | Δ<−0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1-E0 | 0.020354 | -0.000000 | [-0.011545, 0.058466] | 14 | 8 | 22 | 13 | 10 |
| H1v1-E1 | -0.000591 | 0.000000 | [-0.001370, 0.000086] | 19 | 12 | 13 | 0 | 0 |
| H1v1-E0 | 0.019763 | -0.000000 | [-0.011696, 0.057621] | 14 | 8 | 22 | 13 | 10 |

## Protocol notes

- E0: frozen VoxTell; E1: fixed 500-step high-only self-training; H1-v1: same initialization, cases, pseudo-label cache and budget, plus cross-case top-5 relation loss (λ=0.1, temperature=0.1).
- Prompt: `liver cancer`; only decoder Q/K/V LoRA parameters were trainable in E1/H1-v1.
- Evaluation used the repository's fixed sliding-window inference, 192³ patches, 0.5 tile step and Gaussian blending.
- The 44 cases are never used for training or checkpoint selection in this run. Metrics are post-hoc diagnostics.
- Frozen teacher produced 49 usable foreground case prototypes from 59 High cases; pairwise cosine mean/median was 0.9609/0.9692 (p10–p90 0.9166–0.9918), indicating substantial shared feature direction and a potential weak-discrimination limitation for H1-v1.
- Engineering target check (mean Dice ≥0.38): E0=False, E1=False, H1-v1=False.
- Per-case paired metrics: `per_case_metrics.csv`.
