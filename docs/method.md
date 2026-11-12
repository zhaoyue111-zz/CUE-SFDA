# Paper-to-code mapping

## Stage 1: uncertainty-guided partition

For each target patch, the implementation creates K=10 strong photometric
views, mixes randomly selected augmented pairs with a Beta coefficient, and
predicts the K mixed views with the frozen source model. Equations (3)--(5) are
implemented as

```text
mu = mean_k(p_k)
variance = mean_k((p_k - mu)^2)
U = variance / (variance + (mu - delta)^2)
```

The confidence mask is `U <= tau`, with `delta=0.5` and `tau=0.8`. The raw
source pseudo-label is filtered by this mask. Their Dice is the case quality
score, and the fixed MRI threshold `gamma=0.5` determines the high/low branch.
An empty raw pseudo-label has quality zero. The partition is computed once with
the frozen source model and is not allowed to drift during adaptation.

Equation (10) alone gives a perfect score to even one stable foreground voxel.
The 3-D implementation therefore requires a label-free minimum source support
before a case can enter the high-quality branch: 1024 accumulated sliding-window
voxels for liver and 64 for liver cancer. This support guard is not in the paper;
it prevents tiny stable false positives from being treated as reliable masks.

## Stage 2: alternating optimization

For a high-quality case, the frozen source probability and final decoder feature
map form a local affinity graph. Feature affinity and downsampled-image affinity
are combined as `I = If * Ix + If`; the normalized transition is applied four
times. The 2-D paper graph is extended to a memory-safe 3-D 26-neighbour graph.
The refined soft mask supervises VoxTell with balanced BCE and soft Dice.

For a low-quality case, multiple strong views are predicted by the current
target model. Their detached mean is the semantic prototype. A differentiable
view is aligned back to the original coordinates and optimized with Bernoulli
`KL(prototype || prediction)`. Keeping only one differentiable 192-cubed graph
is the memory-safe analogue of equation (23).

The paper's low-frequency attention style transfer is deliberately disabled.
It minimizes a source-BN-statistics loss, but the supplied VoxTell plans specify
`InstanceNorm3d`, which has no source running mean or variance. A second input
z-score is not an equivalent implementation.

## Parameter update and label isolation

Rank-8 LoRA deltas are inserted into Q, K and V of the six active prompt
cross-attention layers (589,824 trainable parameters). Source weights, encoder,
convolutional decoder, text encoder, and input prompts remain unchanged. Each
iteration processes a complete target case and averages its sliding-window
gradients before one optimizer step.

High-quality patches below the configured pseudo-foreground minimum contribute
no gradient. Gradients are normalized by the number of effective patches rather
than by every sliding window, so empty background windows cannot dilute a valid
foreground patch. Background context remains inside every retained 192-cubed
patch. Logs report raw/refined foreground fractions and effective/skipped patch
counts.

The 101 adaptation cases are loaded from image paths only. Ground truth is
available solely to the evaluation function. Periodic evaluation is useful for
this engineering target, but selecting a checkpoint on the 44 reported cases is
not an unbiased SFDA protocol.

## Data boundary

Adaptation and inference use only the configured DWI images. Auxiliary MRI
sequences and external datasets are not supported. Ground-truth masks are read
only by evaluation and never enter pseudo-label generation, sampling, losses,
or parameter updates.

The paper applies its affinity graph at full 2-D resolution. The 3-D VoxTell
implementation must pool the graph to fit memory; therefore the refined target
is unioned with the raw source probability so pooling cannot erase a small
foreground seed. This seed-preservation rule is a documented 3-D engineering
extension and uses no labels.

Cases are sampled from the fixed uncertainty partition without annotations.
For the sparse tumor task, the reliable branch is sampled with probability
0.75 so iterations more often contain a usable source-predicted foreground;
the low-quality consistency branch remains active with probability 0.25.

Version-3 checkpoints store Python, NumPy, CPU/CUDA PyTorch, case-sampler, and
augmentation RNG states. Resuming therefore continues the stochastic sequence
instead of replaying it from the original seed.

To approach the paper's end-to-end target-model optimization without exceeding
11 GB GPU memory, low-rank adapters are applied not only to cross-attention Q/K/V
but also to the text projection and final mask projection. The frozen source
teacher disables all of these adapters, so pseudo-label generation remains
anchored to the original checkpoint.
