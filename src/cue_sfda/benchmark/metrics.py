"""Image-level uncertainty metrics for the Stage 1 benchmark."""
from __future__ import annotations

from itertools import combinations

import numpy as np


EPS = 1e-8


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def hard_dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * int(np.logical_and(a, b).sum()) / denom


def pairwise_hard(masks: np.ndarray) -> float:
    scores = [hard_dice(masks[a], masks[b]) for a, b in combinations(range(len(masks)), 2)]
    return float(np.mean(scores)) if scores else float("nan")


def pairwise_soft(probabilities: np.ndarray) -> float:
    scores = []
    for a, b in combinations(range(len(probabilities)), 2):
        pa, pb = probabilities[a].astype(np.float64), probabilities[b].astype(np.float64)
        denominator = float(np.sum(pa * pa) + np.sum(pb * pb))
        scores.append(1.0 if denominator <= EPS else 2.0 * float(np.sum(pa * pb)) / denominator)
    return float(np.mean(scores)) if scores else float("nan")


def vvc(masks: np.ndarray) -> float:
    counts = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float64)
    mean = float(counts.mean())
    return float(counts.std(ddof=0) / (mean + EPS))


def expected_pairwise_kl(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Mean ordered Bernoulli KL over every view pair, including self pairs."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
    mean_p = p.mean(axis=0)
    mean_log_p = np.log(p).mean(axis=0)
    mean_log_not_p = np.log1p(-p).mean(axis=0)
    # Expand the K^-2 ordered-pair sum so no K×K×D×H×W tensor is formed.
    return (np.mean(p * np.log(p) + (1 - p) * np.log1p(-p), axis=0)
            - mean_p * mean_log_p - (1 - mean_p) * mean_log_not_p)


def uncertainty_maps(original: np.ndarray, views: np.ndarray, threshold: float = 0.5) -> dict[str, np.ndarray]:
    pmean = views.mean(axis=0, dtype=np.float64)
    ee = entropy(views).mean(axis=0)
    maps = {
        "I4_MPC": 1.0 - np.maximum(original, 1.0 - original),
        "I5_Entropy": entropy(original),
        "I6_PE": entropy(pmean),
        "I7_EE": ee,
        "I8_PV": views.astype(np.float64).var(axis=0),
        "I9_MI": np.maximum(entropy(pmean) - ee, 0.0),
        "I10_EPKL": expected_pairwise_kl(views),
    }
    votes = (views > threshold).mean(axis=0)
    maps["I11_VD"] = 1.0 - np.maximum(votes, 1.0 - votes)
    return maps


def aggregate(map_: np.ndarray, valid: np.ndarray, union: np.ndarray) -> tuple[float, int]:
    selected = valid if union is None else (valid & union)
    count = int(selected.sum())
    return (float(np.asarray(map_, dtype=np.float64)[selected].mean()), count) if count else (float("nan"), 0)


def confidence(metric: str, uncertainty: float) -> float:
    if not np.isfinite(uncertainty):
        return float("nan")
    if metric in {"I4_MPC", "I11_VD"}:
        return 1.0 - 2.0 * uncertainty
    if metric == "I8_PV":
        return 1.0 - 4.0 * uncertainty
    if metric in {"I0_CUE_DIC", "I1_Pairwise_Hard", "I2_Pairwise_Soft", "I6_PE", "I7_EE", "I9_MI"}:
        return 1.0 - uncertainty
    if metric in {"I3_VVC", "I10_EPKL", "I13_IPD"}:
        return 1.0 / (1.0 + uncertainty)
    if metric == "I5_Entropy":
        return 1.0 - uncertainty
    if metric == "I12_Chebyshev":
        return 1.0 - uncertainty
    return -uncertainty
