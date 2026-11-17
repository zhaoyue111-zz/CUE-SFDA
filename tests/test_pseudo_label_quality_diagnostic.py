from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_pseudo_label_quality import (  # noqa: E402
    bbox_slices,
    bootstrap_spearman_ci,
    dice_masks,
    image_case_rng,
    soft_dice,
    weak_augment,
)
import torch


def test_empty_prediction_convention_is_explicit():
    empty = np.zeros((3, 4), dtype=bool)
    nonempty = empty.copy()
    nonempty[1, 2] = True
    assert dice_masks(empty, empty, both_empty_zero=True) == 0.0
    assert dice_masks(empty, empty) == 1.0
    assert dice_masks(nonempty, empty, both_empty_zero=True) == 0.0


def test_soft_dice_has_explicit_empty_convention():
    empty = np.zeros((2, 3), dtype=np.float32)
    assert soft_dice(empty, empty) == 1.0


def test_bbox_returns_exact_crop_shape():
    crop = bbox_slices(((1, 4), (2, 7), (0, 3)), (5, 8, 6))
    assert tuple(s.stop - s.start for s in crop) == (3, 5, 3)


def test_weak_augmentation_is_reproducible_and_preserves_geometry():
    image = torch.arange(1 * 7 * 8 * 9, dtype=torch.float32).reshape(1, 7, 8, 9)
    first = weak_augment(image, image_case_rng(2026, "case-1"))
    second = weak_augment(image, image_case_rng(2026, "case-1"))
    assert first.shape == image.shape
    assert torch.equal(first, second)


def test_case_bootstrap_ci_tracks_positive_rank_association():
    x = np.arange(20, dtype=np.float64)
    y = x.copy()
    samples = np.random.default_rng(3).integers(0, len(x), size=(250, len(x)))
    low, high, n = bootstrap_spearman_ci(x, y, samples)
    assert n > 0
    assert low > 0.5
    assert high == 1.0
