from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import t as student_t


def flatten_gradients(gradients: tuple[torch.Tensor | None, ...]) -> torch.Tensor | None:
    """Flatten a parameter-aligned gradient tuple without touching .grad fields."""
    pieces = [value.detach().reshape(-1).float() for value in gradients if value is not None]
    if not pieces:
        return None
    result = torch.cat(pieces)
    if not bool(torch.isfinite(result).all()) or float(result.norm().item()) <= 0.0:
        return None
    return result


def pairwise_gradient_agreement(
    candidate: torch.Tensor, references: list[torch.Tensor], eps: float = 1e-12
) -> tuple[float | None, float | None, int]:
    if not references or not bool(torch.isfinite(candidate).all()):
        return None, None, 0
    candidate = candidate.float()
    norm = candidate.norm()
    if not bool(torch.isfinite(norm)) or float(norm.item()) <= eps:
        return None, None, 0
    values = []
    for reference in references:
        if reference.shape != candidate.shape or not bool(torch.isfinite(reference).all()):
            continue
        ref_norm = reference.float().norm()
        if not bool(torch.isfinite(ref_norm)) or float(ref_norm.item()) <= eps:
            continue
        values.append(float((torch.dot(candidate, reference.float()) / (norm * ref_norm + eps)).item()))
    if not values or not np.isfinite(values).all():
        return None, None, 0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, len(values)


def one_sided_delta_test(
    candidate: torch.Tensor,
    references: list[torch.Tensor],
    learning_rate: float,
    significance_level: float,
) -> dict[str, float | int | bool | None]:
    """First-order loss-change test with a finite-sample Student-t bound."""
    if not 0.0 < significance_level < 0.5:
        raise ValueError("significance_level must be in (0, 0.5)")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    products = []
    if bool(torch.isfinite(candidate).all()):
        for reference in references:
            if reference.shape == candidate.shape and bool(torch.isfinite(reference).all()):
                delta = -float(learning_rate) * float(torch.dot(reference.float(), candidate.float()).item())
                if math.isfinite(delta):
                    products.append(delta)
    count = len(products)
    if count < 2:
        return {"mean": None, "std": None, "upper_bound": None, "count": count, "accepted": False}
    mean = float(np.mean(products))
    std = float(np.std(products, ddof=1))
    critical = float(student_t.ppf(1.0 - significance_level, df=count - 1))
    upper = mean + critical * std / math.sqrt(count)
    finite = all(math.isfinite(value) for value in (mean, std, upper))
    return {"mean": mean if finite else None, "std": std if finite else None,
            "upper_bound": upper if finite else None, "count": count,
            "accepted": bool(finite and upper < 0.0)}
