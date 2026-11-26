# Liver foreground expansion controls

This project isolates seven runs on `stage1-quality-routing`: one `whole liver`
CSS baseline and E0–E5. Each YAML selects exactly one experiment through
`liver_adaptation.experiment`; mechanisms are not combined. All runs retain the
same 101/44 split, LoRA model, optimizer, learning rate, 2,000-iteration cap,
CSS semantic-subspace High acceptance, and Low Dynamic-hard objective. Low
Dynamic-hard starts from its existing frozen-source cache, hard-linked into an
experiment-local cache so later refreshes cannot cross-contaminate runs.

## Methods

- **baseline**: existing CSS High loss and Low Dynamic-hard, with `whole liver`
  as the model prompt.
- **E0**: frozen-source soft probability target on High; no Random Walk, max
  preservation, or multi-prompt High target.
- **E1**: baseline plus the one-sided case-volume ceiling.
- **E2**: baseline plus upper and lower case-volume bounds.
- **E3**: baseline plus a distant, low-Teacher-probability background penalty.
- **E4**: baseline plus separate interior, uncertain-boundary, and distant
  background BCE terms.
- **E5**: baseline plus a confidence-weighted KL anchor to frozen Teacher.

E1/E2 stitch every Student tile's logits with the same Gaussian overlap weights
used for inference, crop back to the full non-padding 3D ROI, and differentiate
the whole-case soft volume. To stay within 11 GB, training first records a
no-gradient full-case pass and computes `dL/d(fused logits)`; then it recomputes
each tile and backpropagates its exact normalized overlap-weighted vector
product. The RNG states are replayed so transformer dropout uses the same masks
for the volume pass and its gradient pass. This is not a patch-volume proxy.

The frozen Teacher cache stores float32 ROI probabilities with the prompt,
checkpoint signature, image identity, spacing, ROI/padded shapes, ROI slice,
patch size, and tile step. Writes are per-case locked and atomic. Distance maps
for E3/E4 use the reoriented array-axis spacing in millimetres; empty Teacher
foregrounds safely disable those terms.

Validation writes per-case Dice/mIoU/precision/recall, student and Teacher
soft volumes, thresholded predicted/GT volumes, and Student/Teacher volume
ratios every 50 iterations. Training summaries contain CSS accept counts,
High/Low effective patch counts, and means of the individual loss terms.
The existing pipeline uses the 44-case split for evaluation, checkpoint
selection, and early stopping. These labels do not enter pseudo-labels, volume
constraints, routing, or constraint-threshold selection. Since checkpoints and
the best experiment are selected using these 44 cases, their Dice is a
validation result, not an independent held-out test estimate.

## Run

From the repository root, start the two-GPU scheduler:

```bash
python scripts/run_liver_expansion_controls.py
```

The scheduler runs at most one job per GPU. It records all validation curves in
`outputs/liver_expansion_control/experiment_records.jsonl`. Runs below Dice
0.75 retain their config, logs, routing/training summaries, and validation CSVs
under `outputs/liver_expansion_control/reports/`; model checkpoints and
temporary predictions are removed. A run at or above 0.75 keeps its full
output directory under `outputs/liver_expansion_control/runs/`.
