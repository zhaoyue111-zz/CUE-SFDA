# Results

Only DWI-only runs are valid. New liver and liver-cancer experiments are saved
under `dwi_only_liver*`. Directories from the earlier auxiliary-PRE evaluation
are invalid under the current data constraint and have been removed.

- `dwi_only_liver_expanded/evaluation`: 44 final liver masks and metrics
  (mean Dice 0.732843).
- `dwi_only_liver_cancer_expanded/evaluation`: 44 final tumor masks and metrics
  (mean Dice 0.311355).

Each final directory also retains DWI-only baseline metrics, the best adapter
checkpoint, the 101-case label-free confidence partition, resolved config, and
run log. Intermediate validation masks and smoke-test outputs were removed.
