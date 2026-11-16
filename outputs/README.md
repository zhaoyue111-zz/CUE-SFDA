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

The progressive liver-cancer CUE-SFDA runs requested in `1.md` are stored under
`liver_cancer_mainline/`. Run one stage at a time so its validation summary can
be reviewed before choosing the next stage:

```bash
CUE_GPU=0 bash scripts/run_liver_cancer_mainline.sh A_C_baseline none
CUE_GPU=0 bash scripts/run_liver_cancer_mainline.sh E_clean_language joint_language
CUE_GPU=0 bash scripts/run_liver_cancer_mainline.sh F_dynamic_hard_language dynamic_hard_language
CUE_GPU=0 bash scripts/run_liver_cancer_mainline.sh G_selective_dynamic_refresh selective_dynamic
CUE_GPU=0 bash scripts/run_liver_cancer_mainline.sh H_selective_dynamic_language selective_dynamic_language
```

The runner resumes from `checkpoints/latest.pt` when present. Each run uses the
same 2,000-iteration limit, 50-iteration validation cadence, and patience of 10.
Summaries are written to `liver_cancer_mainline/summary/` after the completed
stages are analyzed.

The mainline runner reuses the existing frozen-source cross-modal cache. Dynamic
stages start from hard-linked version-0 case files in their own cache directory;
atomic refresh writes then separate those files from the shared source cache.
