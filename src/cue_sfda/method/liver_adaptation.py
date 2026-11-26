from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


EXPERIMENTS = {"baseline", "E0", "E1", "E2", "E3", "E4", "E5"}


def case_volume_penalty(
    student_soft_volume: torch.Tensor,
    teacher_soft_volume: torch.Tensor,
    *,
    experiment: str,
    margin: float,
    shrink_weight: float = 1.0,
    epsilon: float = 1e-6,
    minimum_teacher_volume: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return weighted loss, d(loss)/d(student volume), and safety status."""
    teacher_value = float(teacher_soft_volume.detach().float().item())
    zero = student_soft_volume.detach().float().new_zeros(())
    if not np.isfinite(teacher_value) or teacher_value < float(minimum_teacher_volume):
        return zero, zero, False
    if not 0.0 <= float(margin) < 1.0:
        raise ValueError("volume_constraint.margin must be in [0, 1)")
    ratio = student_soft_volume.float() / (teacher_soft_volume.detach().float() + epsilon)
    upper = F.relu(ratio - (1.0 + float(margin)))
    loss = upper.square()
    if experiment == "E2":
        lower = F.relu((1.0 - float(margin)) - ratio)
        loss = loss + float(shrink_weight) * lower.square()
    elif experiment != "E1":
        raise ValueError("case_volume_penalty is only defined for E1 and E2")
    derivative = torch.autograd.grad(loss, student_soft_volume, retain_graph=False)[0]
    return loss.detach(), derivative.detach(), True


def patch_vjp_coefficient(
    fused_logit_gradient: torch.Tensor,
    normalization: torch.Tensor,
    gaussian_weight: torch.Tensor,
    slicer: tuple[slice, slice, slice],
) -> torch.Tensor:
    """Gradient coefficient for one tile in a Gaussian-fused full volume."""
    return (
        fused_logit_gradient[slicer]
        * gaussian_weight
        / normalization[slicer].clamp_min(1e-6)
    )


def fused_volume_constraint_gradient(
    fused_logits: torch.Tensor,
    teacher_roi_probability: torch.Tensor,
    roi_slicer: tuple[slice, slice, slice],
    *,
    experiment: str,
    margin: float,
    weight: float = 1.0,
    shrink_weight: float = 1.0,
    epsilon: float = 1e-6,
    minimum_teacher_volume: float = 1.0,
) -> tuple[float, torch.Tensor | None, float, bool]:
    """Exact whole-ROI volume loss and gradient w.r.t. fused padded logits."""
    fused = fused_logits.detach().float().requires_grad_(True)
    student_roi_probability = torch.sigmoid(fused[roi_slicer])
    student_volume = student_roi_probability.sum()
    teacher_roi = torch.as_tensor(
        teacher_roi_probability, dtype=torch.float32, device=fused.device
    ).detach()
    expected_shape = tuple(
        int(region.stop) - int(region.start or 0) for region in roi_slicer
    )
    if tuple(teacher_roi.shape) != expected_shape:
        raise ValueError(
            "Teacher ROI and stitched Student ROI shapes differ: "
            f"teacher={tuple(teacher_roi.shape)} student={expected_shape}"
        )
    teacher_volume = teacher_roi.sum()
    loss, derivative, valid = case_volume_penalty(
        student_volume,
        teacher_volume,
        experiment=experiment,
        margin=margin,
        shrink_weight=shrink_weight,
        epsilon=epsilon,
        minimum_teacher_volume=minimum_teacher_volume,
    )
    ratio = float(student_volume.detach().item() / max(teacher_volume.item(), 1e-6))
    if not valid or float(weight) <= 0.0 or float(loss.item()) <= 0.0:
        return float(loss.item()) * float(weight), None, ratio, valid
    roi_gradient = (
        derivative
        * student_roi_probability.detach()
        * (1.0 - student_roi_probability.detach())
        * float(weight)
    )
    gradient = torch.zeros_like(fused)
    gradient[roi_slicer] = roi_gradient
    return float(loss.item()) * float(weight), gradient.detach(), ratio, valid


def bernoulli_kl_map(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    teacher = teacher.detach().float().clamp(1e-6, 1.0 - 1e-6)
    student = student.float().clamp(1e-6, 1.0 - 1e-6)
    return teacher * (teacher.log() - student.log()) + (1.0 - teacher) * (
        (1.0 - teacher).log() - (1.0 - student).log()
    )


def teacher_anchor_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
    *,
    foreground_threshold: float = 0.9,
    background_threshold: float = 0.1,
    boundary_weight: float = 0.1,
) -> torch.Tensor:
    p0 = teacher.detach().float()
    weights = torch.full_like(p0, float(boundary_weight))
    weights[(p0 >= foreground_threshold) | (p0 <= background_threshold)] = 1.0
    return (bernoulli_kl_map(p0, student) * weights).sum() / weights.sum().clamp_min(1.0)


def remote_background_loss(
    student: torch.Tensor,
    remote_mask: torch.Tensor,
) -> torch.Tensor:
    mask = remote_mask.bool()
    if not bool(mask.any()):
        return student.sum() * 0.0
    return F.binary_cross_entropy(
        student.float().clamp(1e-6, 1.0 - 1e-6)[mask],
        torch.zeros_like(student.float()[mask]),
        reduction="mean",
    )


def partitioned_teacher_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
    interior_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    remote_background_mask: torch.Tensor,
    *,
    interior_weight: float = 1.0,
    boundary_weight: float = 0.25,
    background_weight: float = 1.0,
) -> torch.Tensor:
    """Equal-scale mean BCE terms over disjoint trusted 3D regions."""
    p0 = teacher.detach().float().clamp(0.0, 1.0)
    ps = student.float().clamp(1e-6, 1.0 - 1e-6)
    regions = (
        (interior_mask.bool(), float(interior_weight), p0),
        (boundary_mask.bool(), float(boundary_weight), p0),
        (remote_background_mask.bool(), float(background_weight), torch.zeros_like(p0)),
    )
    terms: list[tuple[float, torch.Tensor]] = []
    for mask, weight, target in regions:
        if weight <= 0 or not bool(mask.any()):
            continue
        loss = F.binary_cross_entropy(ps[mask], target[mask], reduction="mean")
        terms.append((weight, loss))
    if not terms:
        return student.sum() * 0.0
    weight_sum = sum(weight for weight, _ in terms)
    return sum(weight * loss for weight, loss in terms) / weight_sum


def build_liver_region_masks(
    probability: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    foreground_threshold: float = 0.5,
    remote_probability_max: float = 0.1,
    remote_distance_mm: float = 15.0,
    boundary_width_mm: float = 5.0,
) -> dict[str, np.ndarray]:
    """Build physical-distance regions from a frozen Teacher ROI prediction."""
    from scipy import ndimage

    p0 = np.asarray(probability, dtype=np.float32)
    if p0.ndim != 3:
        raise ValueError(f"Expected 3D teacher probability, got {p0.shape}")
    if len(spacing) != 3 or any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError(f"Invalid 3D spacing: {spacing}")
    foreground = p0 >= float(foreground_threshold)
    if not foreground.any():
        return {"valid": np.asarray(False)}
    inside_distance = ndimage.distance_transform_edt(foreground, sampling=spacing)
    outside_distance = ndimage.distance_transform_edt(~foreground, sampling=spacing)
    interior = foreground & (inside_distance >= float(boundary_width_mm))
    boundary = (foreground & ~interior) | (
        (~foreground) & (outside_distance < float(boundary_width_mm))
    )
    remote_background = (
        (~foreground)
        & (outside_distance >= float(remote_distance_mm))
        & (p0 <= float(remote_probability_max))
    )
    return {
        "valid": np.asarray(True),
        "interior": interior,
        "boundary": boundary,
        "remote_background": remote_background,
        "inside_distance_mm": inside_distance.astype(np.float32),
        "outside_distance_mm": outside_distance.astype(np.float32),
    }
