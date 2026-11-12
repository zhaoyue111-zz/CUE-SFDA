from __future__ import annotations

from dataclasses import dataclass

import torch


def pseudo_label_quality(
    raw_count: int,
    filtered_count: int,
    minimum_foreground_voxels: int = 1,
) -> float:
    """Dice quality with a label-free minimum support requirement.

    Equation (10) alone assigns a perfect score to a single stable foreground
    voxel.  That is not useful evidence for a high-quality 3-D pseudo-label, so
    volumes below the configured support are kept in the low-quality branch.
    """
    raw_count = int(raw_count)
    filtered_count = int(filtered_count)
    minimum = int(minimum_foreground_voxels)
    if raw_count < 0 or filtered_count < 0:
        raise ValueError("Foreground counts must be non-negative")
    if filtered_count > raw_count:
        raise ValueError("Filtered foreground cannot exceed raw foreground")
    if minimum < 1:
        raise ValueError("Minimum foreground support must be positive")
    if raw_count < minimum:
        return 0.0
    return (2.0 * filtered_count) / (raw_count + filtered_count)


@dataclass
class ChebyshevEstimate:
    raw_probability: torch.Tensor
    mean_probability: torch.Tensor
    standard_deviation: torch.Tensor
    error_bound: torch.Tensor
    reliable_mask: torch.Tensor
    confidence_mask: torch.Tensor
    case_confidence: float


def chebyshev_estimate(
    aligned_probabilities: torch.Tensor,
    raw_probability: torch.Tensor | None = None,
    probability_threshold: float = 0.5,
    bound_threshold: float = 0.8,
    epsilon: float = 1e-8,
) -> ChebyshevEstimate:
    if aligned_probabilities.ndim < 2:
        raise ValueError("View probabilities must include a view dimension")
    if aligned_probabilities.shape[0] < 2:
        raise ValueError("At least two views are required")
    probabilities = aligned_probabilities.float()
    mean = probabilities.mean(dim=0)
    standard_deviation = probabilities.std(dim=0, unbiased=False)
    variance = standard_deviation.square()
    distance = (mean - float(probability_threshold)).square()
    error_bound = variance / (variance + distance + float(epsilon))
    error_bound = torch.nan_to_num(error_bound, nan=1.0, posinf=1.0, neginf=1.0)
    confidence_mask = error_bound <= float(bound_threshold)
    reliable = confidence_mask
    raw = mean if raw_probability is None else raw_probability.float()
    raw_mask = raw > float(probability_threshold)
    filtered_mask = raw_mask & confidence_mask
    raw_count = int(raw_mask.sum().item())
    filtered_count = int(filtered_mask.sum().item())
    # Eq. (10): Dice between the raw pseudo-label and its uncertainty-filtered
    # counterpart. The case-level calibration applies any larger minimum
    # support after accumulating all sliding-window counts.
    case_confidence = pseudo_label_quality(raw_count, filtered_count)
    return ChebyshevEstimate(
        raw_probability=raw,
        mean_probability=mean,
        standard_deviation=standard_deviation,
        error_bound=error_bound,
        reliable_mask=reliable,
        confidence_mask=confidence_mask,
        case_confidence=case_confidence,
    )


class ConfidencePartition:
    def __init__(self, quality_threshold: float, momentum: float) -> None:
        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("Quality threshold must be in [0, 1]")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Confidence EMA momentum must be in [0, 1)")
        self.quality_threshold = float(quality_threshold)
        self.momentum = float(momentum)
        self.scores: dict[str, float] = {}
        self.threshold = self.quality_threshold
        self.high_confidence_names: set[str] = set()

    def initialize(self, scores: dict[str, float]) -> None:
        if not scores:
            raise ValueError("Cannot initialize an empty confidence partition")
        self.scores = {name: float(score) for name, score in scores.items()}
        self._refresh_threshold()

    def update(self, case_name: str, score: float) -> None:
        previous = self.scores.get(case_name, float(score))
        updated = self.momentum * previous + (1.0 - self.momentum) * float(score)
        self.scores[case_name] = updated
        self._refresh_threshold()

    def is_high_confidence(self, case_name: str) -> bool:
        if case_name not in self.scores:
            raise KeyError(f"Case has not been calibrated: {case_name}")
        return case_name in self.high_confidence_names

    def _refresh_threshold(self) -> None:
        self.threshold = self.quality_threshold
        # Eq. (11) uses a fixed modality-dependent gamma, not a dataset
        # quantile. Empty predictions deliberately remain low-quality.
        self.high_confidence_names = {
            name for name, score in self.scores.items()
            if score > self.quality_threshold
        }
