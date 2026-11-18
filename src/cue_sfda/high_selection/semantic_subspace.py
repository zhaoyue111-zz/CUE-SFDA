from __future__ import annotations

from itertools import combinations

import numpy as np

EXTERNAL_CANDIDATE = "__external_candidate__"


def fit_subspace(features: dict[str, np.ndarray], energy: float = 0.9) -> tuple[np.ndarray, int, float]:
    if not 0.0 < energy <= 1.0:
        raise ValueError("spectral_energy must be in (0, 1]")
    if len(features) < 2:
        raise ValueError("at least two reference cases are required")
    arrays = []
    dimensions = set()
    for value in features.values():
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("each case must provide a non-empty [tokens, channels] matrix")
        if not np.isfinite(array).all():
            raise ValueError("semantic features contain NaN/Inf")
        dimensions.add(int(array.shape[1]))
        arrays.append(array)
    if len(dimensions) != 1:
        raise ValueError("semantic feature channel dimensions differ")
    matrix = np.concatenate(arrays, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    matrix = matrix[valid] / norms[valid]
    if matrix.shape[0] < 2:
        raise ValueError("insufficient non-zero semantic feature tokens")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    power = singular.astype(np.float64) ** 2
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0 or not np.isfinite(power).all():
        raise ValueError("invalid semantic feature spectrum")
    rank = int(np.searchsorted(np.cumsum(power) / total, energy, side="left") + 1)
    rank = min(rank, vh.shape[0])
    basis = vh[:rank].T.astype(np.float32)
    if not np.isfinite(basis).all():
        raise ValueError("non-finite semantic basis")
    return basis, rank, float(power[:rank].sum() / total)


def relative_subspace_error(features: np.ndarray, basis: np.ndarray) -> float:
    basis = np.asarray(basis, dtype=np.float32)
    values = np.asarray(features, dtype=np.float32)
    if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] < 1:
        raise ValueError("basis must have shape [channels, rank]")
    if values.ndim != 2 or values.shape[1] != basis.shape[0] or values.shape[0] == 0:
        raise ValueError("candidate features must have shape [tokens, basis channels]")
    if not np.isfinite(values).all() or not np.isfinite(basis).all():
        raise ValueError("non-finite semantic features or basis")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    values = values[valid] / norms[valid]
    if values.size == 0:
        raise ValueError("candidate contains no non-zero feature tokens")
    projection = (values @ basis) @ basis.T
    denominator = float(np.square(values).sum())
    error = float(np.square(values - projection).sum() / max(denominator, 1e-12))
    if not np.isfinite(error):
        raise ValueError("non-finite semantic subspace residual")
    return error


def robust_threshold(errors: list[float], multiplier: float = 3.0) -> float:
    if multiplier < 0:
        raise ValueError("robust MAD multiplier cannot be negative")
    finite = np.asarray(errors, dtype=np.float64)
    if finite.size < 2 or not np.isfinite(finite).all():
        raise ValueError("need at least two finite leave-one-case-out residuals")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    threshold = median + float(multiplier) * max(1.4826 * mad, 1e-6)
    if not np.isfinite(threshold):
        raise ValueError("non-finite robust semantic threshold")
    return threshold


def build_semantic_cache(
    features: dict[str, np.ndarray], energy: float = 0.9, multiplier: float = 3.0
) -> dict[str, dict]:
    """Build full and candidate-excluded CSS evidence once per reference refresh.

    For a candidate absent from the reference bank, use full-bank basis and its
    leave-one-reference-out residual threshold. For a candidate present in the
    bank, use the basis from R\\{candidate} and calculate its threshold from
    pairwise-excluded bases R\\{candidate, heldout}. No candidate can contribute
    to its own basis or threshold.
    """
    case_ids = sorted(features)
    if len(case_ids) < 4:
        raise ValueError("CSS cache requires at least four valid reference cases")
    evidence: dict[str, dict] = {}
    full_basis, full_rank, full_energy = fit_subspace(features, energy)

    single_bases: dict[str, tuple[np.ndarray, int, float]] = {
        omitted: fit_subspace({key: value for key, value in features.items() if key != omitted}, energy)
        for omitted in case_ids
    }
    external_errors = [
        relative_subspace_error(features[heldout], single_bases[heldout][0])
        for heldout in case_ids
    ]
    external_threshold = robust_threshold(external_errors, multiplier)
    evidence[EXTERNAL_CANDIDATE] = {
        "basis": full_basis, "rank": full_rank, "spectral_energy": full_energy,
        "threshold": external_threshold, "reference_case_count": len(case_ids),
    }

    pair_bases: dict[tuple[str, str], tuple[np.ndarray, int, float]] = {}
    for first, second in combinations(case_ids, 2):
        remaining = {key: value for key, value in features.items() if key not in {first, second}}
        pair_bases[(first, second)] = fit_subspace(remaining, energy)
    for candidate in case_ids:
        refs = [case_id for case_id in case_ids if case_id != candidate]
        errors = []
        for heldout in refs:
            key = tuple(sorted((candidate, heldout)))
            basis = pair_bases[key][0]
            errors.append(relative_subspace_error(features[heldout], basis))
        threshold = robust_threshold(errors, multiplier)
        basis, rank, retained = single_bases[candidate]
        evidence[candidate] = {
            "basis": basis, "rank": rank, "spectral_energy": retained,
            "threshold": threshold, "reference_case_count": len(refs),
        }
    return evidence
