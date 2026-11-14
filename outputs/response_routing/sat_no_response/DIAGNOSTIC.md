# SAT external diagnostic on D No-response cases

SAT-Nano (official general SAT checkpoint, paired with the released knowledge text encoder) was run offline with one standard forward per case and the fixed prompt `liver cancer`. No SAT output was used for VoxTell adaptation.

## D gate verification

The D routing file contains 101 training cases: 17 `No-response`, 84 `response-present`. The gate uses the raw VoxTell probability mask at threshold 0.5 before confidence filtering; all 17 No-response cases have exactly zero raw foreground voxels (`no_response_all_full_volume_predictions_empty=true`). No Stable/Unstable update logic was changed.

## Result

| quantity | value |
|---|---:|
| No-response cases | 17 |
| SAT non-empty | 17/17 (100.0%) |
| Dice > 0.1 | 5/17 (29.4%) |
| Dice > 0.3 | 5/17 (29.4%) |
| Dice > 0.5 | 0/17 (0.0%) |
| mean Dice | 0.1068 |
| mean Precision | 0.2716 |
| mean Recall | 0.1148 |
| non-empty SAT with Dice <= 0.1 | 12/17 (70.6%) |

SAT therefore finds foreground in every VoxTell-empty case, but most recovered masks are low-quality false-positive proposals. Five cases reach Dice above 0.3; none exceed 0.5. This supports testing SAT as a proposal only with an additional conservative validation/filtering mechanism, not direct replacement of the current gate.

Per-case results: `diagnostic/per_case.csv`; machine-readable summary: `diagnostic/summary.json` and `diagnostic/summary.csv`.
