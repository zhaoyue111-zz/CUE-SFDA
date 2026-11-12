import torch

from cue_sfda.method.uncertainty import (
    ConfidencePartition,
    chebyshev_estimate,
    pseudo_label_quality,
)


def test_identical_views_are_reliable_away_from_boundary():
    probability = torch.full((1, 1, 4, 4, 4), 0.9)
    views = torch.stack([probability, probability, probability], dim=0)
    estimate = chebyshev_estimate(views, bound_threshold=0.05)
    assert torch.allclose(estimate.error_bound, torch.zeros_like(estimate.error_bound))
    assert estimate.reliable_mask.all()
    assert estimate.case_confidence == 1.0


def test_disagreement_at_boundary_is_unreliable():
    low = torch.full((1, 1, 2, 2, 2), 0.1)
    high = torch.full((1, 1, 2, 2, 2), 0.9)
    estimate = chebyshev_estimate(torch.stack([low, high]), bound_threshold=0.05)
    assert not estimate.reliable_mask.any()
    assert estimate.case_confidence == 0.0


def test_empty_raw_pseudo_label_has_zero_quality():
    background = torch.full((1, 1, 2, 2, 2), 0.01)
    views = torch.stack([background, background, background], dim=0)
    estimate = chebyshev_estimate(
        views,
        bound_threshold=0.05,
    )
    assert estimate.reliable_mask.all()
    assert estimate.confidence_mask.all()
    assert estimate.case_confidence == 0.0


def test_quality_is_dice_between_raw_and_filtered_masks():
    views = torch.full((3, 1, 1, 1, 2), 0.9)
    views[:, :, :, :, 1] = torch.tensor([0.2, 0.8, 0.5]).reshape(3, 1, 1, 1)
    raw = torch.full((1, 1, 1, 1, 2), 0.9)
    estimate = chebyshev_estimate(
        views,
        raw_probability=raw,
        bound_threshold=0.05,
    )
    assert estimate.confidence_mask.sum() == 1
    assert abs(estimate.case_confidence - 2 / 3) < 1e-6


def test_confidence_partition_uses_fixed_quality_threshold():
    partition = ConfidencePartition(quality_threshold=0.5, momentum=0.9)
    partition.initialize({"a": 0.1, "b": 0.5, "c": 0.9})
    assert partition.threshold == 0.5
    assert not partition.is_high_confidence("a")
    assert not partition.is_high_confidence("b")
    assert partition.is_high_confidence("c")


def test_zero_quality_cases_remain_low_quality():
    partition = ConfidencePartition(quality_threshold=0.5, momentum=0.9)
    partition.initialize({name: 0.0 for name in "abcdef"})
    selected = [name for name in "abcdef" if partition.is_high_confidence(name)]
    assert not selected


def test_tiny_stable_pseudo_label_is_not_high_quality():
    assert pseudo_label_quality(1, 1, minimum_foreground_voxels=64) == 0.0
    assert pseudo_label_quality(63, 63, minimum_foreground_voxels=64) == 0.0
    assert pseudo_label_quality(64, 64, minimum_foreground_voxels=64) == 1.0
