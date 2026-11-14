# SAT proposal vs frozen VoxTell soft response

The 17 existing D No-response cases were re-run with the frozen VoxTell model using the standard single-prompt (`liver cancer`) tiled inference. No augmentation, adaptation, or GT-dependent response calculation was used. VoxTell probabilities were geometrically transformed to SAT's RAS/1x1x3/cropped grid before sampling inside each SAT mask; GT was used only for the post-hoc SAT Dice label (`Dice > 0.1`).

## Main result

| response/proposal group | n | mean SAT Dice | mean VoxTell top-1% response |
|---|---:|---:|---:|
| effective (`SAT Dice > 0.1`) | 5 | 0.3425 | 0.0008368 |
| ineffective (`SAT Dice <= 0.1`) | 12 | 0.0046 | 9.48e-13 |

The effective mean is dominated by one case (`0017174204`); three other effective cases have essentially zero residual VoxTell response. Thus VoxTell soft response is not a universal validator, although it identifies one useful proposal strongly.

## Correlation/classification (n=17; exploratory)

| metric | Pearson r | Spearman r | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| SAT volume | 0.598 | 0.778 | 0.967 | 0.943 |
| VoxTell mean | 0.252 | 0.313 | 0.833 | 0.673 |
| VoxTell max | 0.252 | 0.409 | 0.867 | 0.759 |
| VoxTell top 1% mean | 0.252 | 0.387 | 0.850 | 0.739 |
| VoxTell top 5% mean | 0.252 | 0.361 | 0.850 | 0.739 |

The volume signal is stronger than the soft-response signals in this small sample. A fixed exploratory rule `SAT nonempty + VoxTell max > 0.01` selects 1/17 case and that case is effective; this is not a fitted or training threshold and has only 20% recall of effective proposals. Top-k response thresholds at 0.01 select no cases.

Per-case data: `per_case.csv`; correlations: `correlations.csv`; AUROC/AUPRC: `classification_scores.csv`; fixed diagnostic thresholds: `fixed_threshold_sweep.csv`; plots: `scatter_dice_vs_top1.png`, `scatter_dice_vs_volume.png`.
