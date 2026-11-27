# CUE-SFDA hierarchical dynamic uncertainty

- Branch: `hierarchical-dynamic-sfda`
- Training uses two GPUs concurrently with isolated output/cache directories.
- M1: all cases sampled uniformly; High and Low strategies both backpropagated per patch; no PE weighting.
- M0 aligned control: historical routed Dynamic-hard training protocol, new isolated output for a same-budget comparison.
- Both configurations use 2,000 iterations, seed 42, 192³ patches, LR 1e-5, and one Dynamic-hard refresh cycle.
- Checkpoints are written every 50 iterations; resume commands are in `experiment_queue.json`.
- No target GT is used in training or sampling. The configured GT path is used only for validation.

## Current status

Running: M1 on physical GPU 0 (execution session 93334, PID 2678705) and aligned Dynamic-hard control on physical GPU 1 (execution session 75153, PID 2679494). Both are building isolated version-0 caches from the frozen source teacher; no training iteration has started yet.

- Fixed PE reliability cache: complete for all 101 training cases using existing Stage-1 probabilities; entropy uses log2, 0.5 image + 0.5 prompt, padding aligned to the training volume. This required no inference.
- Empty original predictions detected from that cache are tagged so M2 can suppress extra all-background pseudo supervision.
- M2 config and weighted-loss code are prepared; it remains gated on review of M1 validation results.
