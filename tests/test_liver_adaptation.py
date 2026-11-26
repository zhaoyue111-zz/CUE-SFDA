import numpy as np
import torch

from cue_sfda.method.liver_adaptation import (
    build_liver_region_masks,
    case_volume_penalty,
    fused_volume_constraint_gradient,
    patch_vjp_coefficient,
)


def test_e1_volume_penalty_has_correct_gradient_and_margin():
    teacher = np.full((1, 1, 1), 1.0, dtype=np.float32)
    # Four identical voxels at sigmoid(0)=0.5 gives ratio 2 against V0=1.
    fused = torch.zeros((4, 1, 1), requires_grad=True)
    loss, gradient, ratio, valid = fused_volume_constraint_gradient(
        fused,
        teacher,
        (slice(0, 1), slice(0, 1), slice(0, 1)),
        experiment="E1",
        margin=0.1,
        weight=2.0,
    )
    assert valid
    assert ratio == 0.5  # The selected ROI is one voxel, below the upper bound.
    assert loss == 0.0
    assert gradient is None

    roi = (slice(0, 4), slice(0, 1), slice(0, 1))
    teacher = np.full((4, 1, 1), 0.25, dtype=np.float32)
    loss, gradient, ratio, valid = fused_volume_constraint_gradient(
        fused,
        teacher,
        roi,
        experiment="E1",
        margin=0.1,
        weight=2.0,
    )
    assert valid and ratio == 2.0 and gradient is not None
    assert loss > 0.0 and torch.count_nonzero(gradient) == 4
    assert torch.allclose(gradient, torch.full_like(gradient, 0.9), atol=1e-5)


def test_e2_penalizes_shrinkage_and_skips_empty_teacher():
    student_volume = torch.tensor(70.0, requires_grad=True)
    teacher_volume = torch.tensor(100.0)
    loss, derivative, valid = case_volume_penalty(
        student_volume,
        teacher_volume,
        experiment="E2",
        margin=0.1,
        shrink_weight=2.0,
    )
    assert valid and loss.item() > 0.0
    assert derivative.item() < 0.0
    _, _, valid = case_volume_penalty(
        student_volume,
        torch.tensor(0.0),
        experiment="E2",
        margin=0.1,
        minimum_teacher_volume=1.0,
    )
    assert not valid


def test_overlapping_tile_vjp_coefficients_reconstruct_global_gradient():
    fused_gradient = torch.arange(1.0, 6.0).view(5, 1, 1)
    gaussian = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    first = (slice(0, 3), slice(0, 1), slice(0, 1))
    second = (slice(2, 5), slice(0, 1), slice(0, 1))
    normalization = torch.zeros_like(fused_gradient)
    normalization[first] += gaussian
    normalization[second] += gaussian
    reconstructed = torch.zeros_like(fused_gradient)
    for slicer in (first, second):
        coefficient = patch_vjp_coefficient(
            fused_gradient, normalization, gaussian, slicer
        )
        reconstructed[slicer] += coefficient
    assert torch.allclose(reconstructed, fused_gradient)


def test_physical_liver_regions_use_anisotropic_spacing_and_skip_empty():
    probability = np.zeros((7, 25, 25), dtype=np.float32)
    probability[3, 12, 12] = 1.0
    result = build_liver_region_masks(
        probability,
        spacing=(7.8, 1.0, 1.0),
        foreground_threshold=0.5,
        remote_probability_max=0.1,
        remote_distance_mm=10.0,
        boundary_width_mm=2.0,
    )
    assert bool(result["valid"].item())
    assert result["outside_distance_mm"][4, 12, 12] == 7.8
    assert result["remote_background"][5, 12, 12]
    empty = build_liver_region_masks(
        np.zeros((3, 4, 5), dtype=np.float32),
        spacing=(7.8, 1.0, 1.0),
    )
    assert not bool(empty["valid"].item())
