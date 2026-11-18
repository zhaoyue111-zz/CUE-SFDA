from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
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
    """Extract foreground-indexed d5_pre_head tokens under frozen source LoRA."""
    with torch.no_grad(), lora_disabled(model):
        with autocast():
            logits, features = model(image, embedding, return_features=True)
        mask = torch.sigmoid(logits.float()) >= float(probability_threshold)
        if not bool(mask.any()):
            return np.empty((0, int(features.shape[1])), dtype=np.float32)
        tokens = features.float().permute(0, 2, 3, 4, 1)[mask[0, 0]].detach().cpu().numpy()
    if not len(tokens):
        return np.empty((0, int(features.shape[1])), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    if len(tokens) > int(max_tokens):
        tokens = tokens[rng.choice(len(tokens), size=int(max_tokens), replace=False)]
    return tokens.astype(np.float32, copy=False)
