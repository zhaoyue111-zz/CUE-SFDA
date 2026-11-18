from __future__ import annotations

import torch

from cue_sfda.high_selection.statistics import one_sided_delta_test


def decide(candidate: torch.Tensor, references: list[torch.Tensor], learning_rate: float, significance_level: float) -> dict:
    result = one_sided_delta_test(candidate, references, learning_rate, significance_level)
    accepted = bool(result["accepted"])
    return {
        "accepted": accepted,
        "reason": "negative_delta_upper_bound" if accepted else "delta_test_rejected_or_invalid",
        "estimated_delta_mean": result["mean"],
        "estimated_delta_std": result["std"],
        "estimated_delta_upper_bound": result["upper_bound"],
        "reference_count": result["count"],
    }
