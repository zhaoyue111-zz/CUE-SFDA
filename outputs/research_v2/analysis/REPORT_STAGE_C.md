# Stage C: frozen High memory transfer to Low

The 44-case held-out set has been reused in prior exploration; these are fixed-iteration labeled development diagnostics, not an untouched final test. GT was not used in training or checkpoint selection.

| Run | Dice | Precision | Recall | IoU |
|---|---:|---:|---:|---:|
| E3 | 0.312517 | 0.346587 | 0.439901 | 0.238338 |
| E4 | 0.294473 | 0.341192 | 0.371068 | 0.222650 |

## Paired E4 − E3 Dice

- Mean Δ: -0.018044; median Δ: 0.000000.
- 95% paired bootstrap CI: [-0.048678, 0.011501].
- Improved / unchanged / declined: 26 / 2 / 16.
- Δ>0.02 / Δ<−0.02: 10 / 12.

## Protocol notes

- C_joint fixed train split: 101 train cases (59 High, 42 Low); 44 held-out cases.
- Both runs use the same frozen-source High pseudo-label cache, 1000 updates, seed, Q/K/V LoRA set, and raw-image frozen-source Bernoulli-KL Low baseline.
- E4 adds only the foreground-masked D5 cosine relation to frozen High case prototypes (weight 0.1). If a Low frozen-source patch has no foreground, the relation term is skipped; no GT or response-based gate is used.
- The repository's default low branch uses augmented-view consistency. For this plan's raw-image-only requirement, E3/E4 used the explicit opt-in raw frozen-source KL path; default behavior was not changed.
- Evaluation used 192³ sliding windows, tile step 0.5, and Gaussian blending.
- Per-case paired results: `per_case_metrics_stage_c.csv`.
