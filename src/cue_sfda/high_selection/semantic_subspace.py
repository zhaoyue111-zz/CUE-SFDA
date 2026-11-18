from __future__ import annotations

import numpy as np


def fit_subspace(features: dict[str, np.ndarray], energy: float = 0.9) -> tuple[np.ndarray, int, float]:
    if not 0.0 < energy <= 1.0:
        raise ValueError("spectral_energy must be in (0, 1]")
    arrays = [np.asarray(value, dtype=np.float32) for value in features.values()]
    arrays = [value.reshape(-1, value.shape[-1]) for value in arrays if value.size]
    if len(arrays) < 2 or not arrays or len({a.shape[1] for a in arrays}) != 1:
        raise ValueError("at least two compatible non-empty case feature sets are required")
    matrix = np.concatenate(arrays, axis=0)
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if matrix.shape[0] < 2:
        raise ValueError("insufficient finite semantic feature tokens")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix[norms[:, 0] > 1e-8] / norms[norms[:, 0] > 1e-8]
    if matrix.shape[0] < 2:
        raise ValueError("insufficient non-zero semantic feature tokens")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    power = singular.astype(np.float64) ** 2
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid semantic feature spectrum")
    rank = int(np.searchsorted(np.cumsum(power) / total, energy, side="left") + 1)
    rank = min(rank, vh.shape[0])
    return vh[:rank].T.astype(np.float32), rank, float(power[:rank].sum() / total)


def relative_subspace_error(features: np.ndarray, basis: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float32).reshape(-1, basis.shape[0])
    values = values[np.isfinite(values).all(axis=1)]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values[norms[:, 0] > 1e-8] / norms[norms[:, 0] > 1e-8]
    if values.size == 0 or basis.ndim != 2 or basis.shape[0] != values.shape[1]:
        raise ValueError("empty or incompatible semantic features")
    projection = (values @ basis) @ basis.T
    denominator = float(np.square(values).sum())
    error = float(np.square(values - projection).sum() / max(denominator, 1e-12))
    if not np.isfinite(error):
        raise ValueError("non-finite semantic subspace residual")
    return error


def robust_threshold(errors: list[float], multiplier: float = 3.0) -> float:
    finite = np.asarray([x for x in errors if np.isfinite(x)], dtype=np.float64)
    if finite.size < 2:
        raise ValueError("need at least two leave-one-case-out residuals")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    # A floor avoids zero-MAD rejecting every non-identical candidate.
    return median + float(multiplier) * max(1.4826 * mad, 1e-6)
