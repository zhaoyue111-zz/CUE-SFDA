from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cue_sfda.modeling.lora import lora_disabled


def foreground_d5_tokens(
    model: nn.Module,
    image: torch.Tensor,
    embedding: torch.Tensor,
    probability_threshold: float,
    max_tokens: int,
    seed: int,
    autocast: Callable,
) -> np.ndarray:
    """Return sampled foreground d5_pre_head tokens with checked 3D alignment.

    VoxTell's current ``return_features=True`` path returns ``d5_pre_head`` as
    [B,C,D,H,W], immediately before the final mask einsum. Its native path has
    the same spatial grid as logits; nearest-neighbor mask resampling handles
    future valid output-grid differences without flatten/reshape misalignment.
    ``max_tokens`` is explicit seeded sampling, never silent prefix truncation.
    """
    if image.ndim != 5:
        raise ValueError(f"expected image [B,C,D,H,W], got {tuple(image.shape)}")
    if image.shape[0] != 1:
        raise ValueError("High semantic feature extraction expects batch size one")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    with torch.no_grad(), lora_disabled(model):
        with autocast():
            logits, features = model(image, embedding, return_features=True)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 5:
            raise ValueError("VoxTell logits must have shape [B,1,D,H,W]")
        if not isinstance(features, torch.Tensor) or features.ndim != 5:
            raise ValueError("VoxTell d5_pre_head features must have shape [B,C,D,H,W]")
        if (
            logits.shape[0] != image.shape[0]
            or features.shape[0] != image.shape[0]
            or logits.shape[1] != 1
        ):
            raise ValueError(
                f"logit/feature batch or prompt mismatch: logits={tuple(logits.shape)}, "
                f"features={tuple(features.shape)}"
            )
        if features.shape[1] < 1 or any(size < 1 for size in features.shape[2:]):
            raise ValueError(f"invalid D5 feature shape: {tuple(features.shape)}")
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(features).all()):
            raise ValueError("non-finite VoxTell logits or D5 features")
        binary_mask = torch.sigmoid(logits.float()) >= float(probability_threshold)
        if tuple(binary_mask.shape[2:]) != tuple(features.shape[2:]):
            binary_mask = F.interpolate(
                binary_mask.float(), size=features.shape[2:], mode="nearest"
            ).bool()
        # Index batch explicitly before applying the aligned 3D mask.
        selected_mask = binary_mask[0, 0]
        if not bool(selected_mask.any()):
            return np.empty((0, int(features.shape[1])), dtype=np.float32)
        tokens = features.float().permute(0, 2, 3, 4, 1)[0][selected_mask]
        if tokens.numel() == 0:
            return np.empty((0, int(features.shape[1])), dtype=np.float32)
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("non-finite selected D5 foreground features")
        tokens = tokens.detach().cpu().numpy()
    rng = np.random.default_rng(int(seed))
    if len(tokens) > max_tokens:
        selected = rng.choice(len(tokens), size=max_tokens, replace=False)
        tokens = tokens[selected]
    return np.asarray(tokens, dtype=np.float32)
