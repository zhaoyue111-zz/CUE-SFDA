from __future__ import annotations

import torch
import torch.nn.functional as F


def confidence_weight_map(
    confidence: torch.Tensor,
    function: str = "linear",
    floor: float = 0.3,
    scale: float = 0.7,
    power: float = 1.0,
) -> torch.Tensor:
    """Map voxel confidence in [0, 1] to strictly positive supervision weights."""
    confidence = confidence.detach().float().clamp(0.0, 1.0)
    floor = float(floor)
    scale = float(scale)
    if floor <= 0.0 or scale < 0.0:
        raise ValueError("confidence weighting requires floor > 0 and scale >= 0")
    if function == "linear":
        transformed = confidence
    elif function == "power":
        if float(power) <= 0.0:
            raise ValueError("confidence weighting power must be positive")
        transformed = confidence.pow(float(power))
    else:
        raise ValueError(f"Unsupported confidence weighting function: {function}")
    return floor + scale * transformed


def masked_binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy(
        probability.clamp(1e-6, 1.0 - 1e-6),
        target.float(),
        reduction="none",
    )
    return (loss * weight.float()).sum() / weight.sum().clamp_min(1.0)


def balanced_masked_binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """BCE with equal foreground/background mass inside the reliable region."""
    loss = F.binary_cross_entropy(
        probability.float().clamp(1e-6, 1.0 - 1e-6),
        target.float(),
        reduction="none",
    )
    weight = weight.float()
    foreground = (target >= float(threshold)).float() * weight
    background = (target < float(threshold)).float() * weight
    terms = []
    if bool(foreground.sum() > 0):
        terms.append((loss * foreground).sum() / foreground.sum())
    if bool(background.sum() > 0):
        terms.append((loss * background).sum() / background.sum())
    if not terms:
        return loss.sum() * 0.0
    return torch.stack(terms).mean()


def masked_soft_dice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    probability = probability.float()
    target = target.float()
    weight = weight.float()
    intersection = (probability * target * weight).sum()
    denominator = ((probability + target) * weight).sum()
    return 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)


def consistency_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    foreground_weight: float = 1.0,
) -> torch.Tensor:
    target = teacher.detach()
    weight = 1.0 + (float(foreground_weight) - 1.0) * target
    squared_error = (student - target).square()
    return (squared_error * weight).sum() / weight.sum().clamp_min(1.0)


def balanced_consistency_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Give sparse foreground and background equal influence."""
    target = teacher.detach().float()
    squared_error = (student.float() - target).square()
    foreground = target >= float(threshold)
    background = ~foreground
    terms = []
    if bool(foreground.any()):
        terms.append(squared_error[foreground].mean())
    if bool(background.any()):
        terms.append(squared_error[background].mean())
    return torch.stack(terms).mean()


def bernoulli_kl_divergence(
    prototype: torch.Tensor,
    prediction: torch.Tensor,
    balanced: bool = False,
    threshold: float = 0.5,
) -> torch.Tensor:
    """KL(prototype || prediction) for a two-class Bernoulli distribution."""
    target = prototype.detach().float().clamp(1e-6, 1.0 - 1e-6)
    student = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    divergence = target * (target.log() - student.log())
    divergence = divergence + (1.0 - target) * (
        (1.0 - target).log() - (1.0 - student).log()
    )
    if not balanced:
        return divergence.mean()
    foreground = target >= float(threshold)
    terms = []
    if bool(foreground.any()):
        terms.append(divergence[foreground].mean())
    if bool((~foreground).any()):
        terms.append(divergence[~foreground].mean())
    return torch.stack(terms).mean()


def binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp(1e-6, 1.0 - 1e-6)
    entropy = -probability * probability.log()
    entropy = entropy - (1.0 - probability) * (1.0 - probability).log()
    return entropy.mean()
