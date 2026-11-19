# CSS Liver experiments

This project keeps Liver-specific training changes in a separate experiment
area. The VoxTell model and LoRA architecture are unchanged. `random_walk_alpha`
now anchors the iterative random walk to its initial teacher score; alpha zero
reproduces the previous unanchored behavior. Liver trial configs use average
score pooling and nonzero anchoring to reduce the spread of pseudo-mask edges.
The consensus-anchor trial also trains High samples from the existing
multi-prompt, multi-view Dynamic-hard cache, weighted by voxel confidence.

`run_parallel.py` starts the first two configs on GPUs 0 and 1. Each run writes into a
temporary staging directory. Runs whose best validation Dice is below 0.75 are
deleted after their validation curve, configuration, exit status, and failure
tail are appended to `experiments.jsonl`. Runs reaching 0.75 are moved to
`results/` with their checkpoints and logs.

The run uses validation labels for checkpoint selection and reporting, as the
existing trainer does. No validation labels enter pseudo-label generation.

## Results so far

The current target is **not reached**. The frozen `liver` baseline scores
0.7328 mean Dice over 44 held-out cases; `whole liver` scores 0.7401. Eight
CSS training variants have been run. The best adapted checkpoint scored
0.7404 (prompt `whole liver`, anchor strength 0.5, iteration 150). The runner
deleted all below-target checkpoints and retained configurations and
validation curves in `experiments.jsonl`.

Additional inference diagnostics also fell short: `whole liver` was the best
single prompt at 0.7401; prompt binary voting/union peaked at 0.7381; keeping
the largest connected component scored 0.7346. These are validation-set
diagnostics and not claims of a held-out improvement. Threshold checks for
`whole liver` scored 0.7392, 0.7397, 0.7404, and 0.7407 at thresholds 0.40,
0.45, 0.55, and 0.60. The best threshold was selected on this same evaluation
split, so it is a diagnostic figure rather than an independent test result.
Failed predictions and model artifacts were removed.

The main failure pattern is foreground expansion during adaptation: recall
increases while precision falls, so later checkpoints do not improve Dice.
The project keeps the best validation checkpoint and stops each variant after
three validations without improvement. Current evidence does not support
claiming stable updates above 0.75 yet. Lowering the learning rate did not
prevent the decline; prompt-consensus updates also did not improve over their
initial checkpoint.
