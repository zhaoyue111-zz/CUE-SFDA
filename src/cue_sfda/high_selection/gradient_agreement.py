from __future__ import annotations

import torch

from cue_sfda.high_selection.statistics import pairwise_gradient_agreement


def decide(candidate: torch.Tensor, references: list[torch.Tensor]) -> dict:
    mean, std, count = pairwise_gradient_agreement(candidate, references)
    accepted = mean is not None and mean > 0.0
    return {
        "accepted": bool(accepted),
        "reason": "positive_mean_gradient_agreement" if accepted else "nonpositive_or_invalid_gradient_agreement",
        "gradient_agreement_mean": mean,
        "gradient_agreement_std": std,
        "reference_count": count,
    }
