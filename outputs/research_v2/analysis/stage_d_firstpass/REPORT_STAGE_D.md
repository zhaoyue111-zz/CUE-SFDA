# Stage D: 50% decoder Q/K/V group routing

Fixed-iteration 1000-step diagnostics on the previously explored 44-case held-out development set. It is not an untouched final test, and GT was not used during training or checkpoint choice.

## Held-out performance

| Run | Dice | Precision | Recall | IoU | Mean paired ΔDice vs E3 | Median ΔDice | 95% paired bootstrap CI | Improved / same / declined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3_R0 | 0.312517 | 0.346587 | 0.439901 | 0.238338 | — | — | — | — |
| E6_fixed50 | 0.297809 | 0.337046 | 0.377139 | 0.225075 | -0.014708 | 0.000000 | [-0.042939, 0.013724] | 25 / 2 / 17 |
| E7_random50 | 0.295436 | 0.339011 | 0.372316 | 0.223508 | -0.017081 | 0.000000 | [-0.047986, 0.013331] | 26 / 2 / 16 |
| E8_norm50 | 0.294349 | 0.340672 | 0.371115 | 0.222514 | -0.018168 | 0.000000 | [-0.048016, 0.011217] | 26 / 2 / 16 |
| E9_cosine50 | 0.302982 | 0.331673 | 0.392321 | 0.229823 | -0.009535 | 0.000000 | [-0.032049, 0.013227] | 23 / 2 / 19 |

The bootstrap resamples the 44 paired case-level Dice differences (10,000 replicates, seed 42). These intervals describe this reused development set and do not replace seed replication.

## Parameter routing audit

All strategies route only Low case updates. Each Low update applies the mask to the averaged case gradient; this is case-update-adaptive, not per-patch sample-adaptive. The 18 groups are six decoder layers × Q/K/V, with 9 selected per Low update. High updates continue to update all groups.

| Run | Low updates | Mean selected groups/update | Group | Selected | Frequency | Mean score |
|---|---:|---:|---|---:|---:|---:|
| E6_fixed50 | 239 | 9.00 | layer0_K | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer0_Q | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer0_V | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer1_K | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer1_Q | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer1_V | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer2_K | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer2_Q | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer2_V | 239 | 1.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer3_K | 0 | 0.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer3_Q | 0 | 0.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer3_V | 0 | 0.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer4_K | 0 | 0.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer4_Q | 0 | 0.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer4_V | 0 | 0.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer5_K | 0 | 0.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer5_Q | 0 | 0.000 | 0.000000 |
| E6_fixed50 | 239 | 9.00 | layer5_V | 0 | 0.000 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer0_K | 131 | 0.548 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer0_Q | 118 | 0.494 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer0_V | 134 | 0.561 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer1_K | 106 | 0.444 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer1_Q | 126 | 0.527 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer1_V | 130 | 0.544 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer2_K | 106 | 0.444 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer2_Q | 109 | 0.456 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer2_V | 121 | 0.506 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer3_K | 121 | 0.506 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer3_Q | 122 | 0.510 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer3_V | 116 | 0.485 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer4_K | 116 | 0.485 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer4_Q | 102 | 0.427 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer4_V | 133 | 0.556 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer5_K | 117 | 0.490 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer5_Q | 131 | 0.548 | 0.000000 |
| E7_random50 | 239 | 9.00 | layer5_V | 112 | 0.469 | 0.000000 |
| E8_norm50 | 239 | 9.00 | layer0_K | 31 | 0.130 | nan |
| E8_norm50 | 239 | 9.00 | layer0_Q | 116 | 0.485 | nan |
| E8_norm50 | 239 | 9.00 | layer0_V | 230 | 0.962 | nan |
| E8_norm50 | 239 | 9.00 | layer1_K | 30 | 0.126 | nan |
| E8_norm50 | 239 | 9.00 | layer1_Q | 61 | 0.255 | nan |
| E8_norm50 | 239 | 9.00 | layer1_V | 179 | 0.749 | nan |
| E8_norm50 | 239 | 9.00 | layer2_K | 24 | 0.100 | nan |
| E8_norm50 | 239 | 9.00 | layer2_Q | 86 | 0.360 | nan |
| E8_norm50 | 239 | 9.00 | layer2_V | 232 | 0.971 | nan |
| E8_norm50 | 239 | 9.00 | layer3_K | 45 | 0.188 | nan |
| E8_norm50 | 239 | 9.00 | layer3_Q | 110 | 0.460 | nan |
| E8_norm50 | 239 | 9.00 | layer3_V | 217 | 0.908 | nan |
| E8_norm50 | 239 | 9.00 | layer4_K | 86 | 0.360 | nan |
| E8_norm50 | 239 | 9.00 | layer4_Q | 118 | 0.494 | nan |
| E8_norm50 | 239 | 9.00 | layer4_V | 187 | 0.782 | nan |
| E8_norm50 | 239 | 9.00 | layer5_K | 38 | 0.159 | nan |
| E8_norm50 | 239 | 9.00 | layer5_Q | 133 | 0.556 | nan |
| E8_norm50 | 239 | 9.00 | layer5_V | 228 | 0.954 | nan |
| E9_cosine50 | 239 | 9.00 | layer0_K | 119 | 0.498 | nan |
| E9_cosine50 | 239 | 9.00 | layer0_Q | 121 | 0.506 | nan |
| E9_cosine50 | 239 | 9.00 | layer0_V | 116 | 0.485 | nan |
| E9_cosine50 | 239 | 9.00 | layer1_K | 130 | 0.544 | nan |
| E9_cosine50 | 239 | 9.00 | layer1_Q | 136 | 0.569 | nan |
| E9_cosine50 | 239 | 9.00 | layer1_V | 117 | 0.490 | nan |
| E9_cosine50 | 239 | 9.00 | layer2_K | 132 | 0.552 | nan |
| E9_cosine50 | 239 | 9.00 | layer2_Q | 123 | 0.515 | nan |
| E9_cosine50 | 239 | 9.00 | layer2_V | 112 | 0.469 | nan |
| E9_cosine50 | 239 | 9.00 | layer3_K | 113 | 0.473 | nan |
| E9_cosine50 | 239 | 9.00 | layer3_Q | 106 | 0.444 | nan |
| E9_cosine50 | 239 | 9.00 | layer3_V | 110 | 0.460 | nan |
| E9_cosine50 | 239 | 9.00 | layer4_K | 118 | 0.494 | nan |
| E9_cosine50 | 239 | 9.00 | layer4_Q | 118 | 0.494 | nan |
| E9_cosine50 | 239 | 9.00 | layer4_V | 123 | 0.515 | nan |
| E9_cosine50 | 239 | 9.00 | layer5_K | 120 | 0.502 | nan |
| E9_cosine50 | 239 | 9.00 | layer5_Q | 116 | 0.485 | nan |
| E9_cosine50 | 239 | 9.00 | layer5_V | 121 | 0.506 | nan |

## Protocol and interpretation limits

- E5/R0 reuses E3 raw Low baseline: same frozen High cache and raw frozen-source Bernoulli-KL Low objective; all Q/K/V groups update. E4 was excluded from R0 because it lowered mean Dice and Recall versus E3.
- E6 selects the first nine sorted groups; E7 selects a seeded random nine; E8 selects the nine groups with largest Low-gradient norms; E9 selects the nine groups with largest cosine scores against a detached independent frozen-cache High reference gradient.
- The High reference is one independent effective High patch per Low update, cycled over available patches. This is a compute-limited approximation to a multi-batch High-set gradient estimate; the reference is a target-domain consensus, not ground truth.
- Unselected gradients are set to None before AdamW.step, preventing momentum and weight decay from moving those groups. The CSV records each selected group and its score.
- A single seed and 1000 updates are exploratory. E9's extra reference forward also changes the stochastic forward-call sequence; group routing results need multi-seed replication before a method claim.
- Fixed/random/norm/cosine variants are compared with the same 50% active-group budget. Final adapter L2 offsets relative to the E3/R0 checkpoint are reported separately from routing frequencies; they do not equal the cumulative optimizer step length.
- Per-case metrics: `per_case_metrics_stage_d.csv`; summary: `summary_stage_d.csv`; routing frequency: `routing_frequency_stage_d.csv`; adapter offsets: `parameter_offsets_stage_d.csv`.
