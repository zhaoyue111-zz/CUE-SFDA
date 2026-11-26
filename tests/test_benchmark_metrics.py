import numpy as np

from cue_sfda.benchmark.metrics import (
    aggregate,
    expected_pairwise_kl,
    hard_dice,
    pairwise_hard,
    pairwise_soft,
    uncertainty_maps,
    vvc,
)


def test_empty_hard_dice_and_pairwise_scores():
    empty = np.zeros((2, 2), dtype=bool)
    full = np.ones((2, 2), dtype=bool)
    assert hard_dice(empty, empty) == 1.0
    assert hard_dice(empty, full) == 0.0
    assert pairwise_hard(np.stack([empty, empty])) == 1.0
    assert pairwise_soft(np.zeros((2, 2, 2), dtype=np.float32)) == 1.0


def test_vvc_uses_population_standard_deviation_and_zero_case():
    masks = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]]], dtype=bool)
    assert np.isclose(vvc(masks), 1.0)
    assert vvc(np.zeros_like(masks)) == 0.0


def test_expected_pairwise_kl_matches_ordered_pair_definition():
    p = np.array([[[0.2, 0.6]], [[0.8, 0.4]], [[0.5, 0.5]]])
    actual = expected_pairwise_kl(p)
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    direct = []
    for a in clipped:
        for b in clipped:
            direct.append(a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b)))
    assert np.allclose(actual, np.mean(direct, axis=0))


def test_uncertainty_maps_and_empty_aggregation():
    original = np.full((2, 2), 0.5, dtype=np.float32)
    views = np.stack([np.zeros((2, 2)), np.ones((2, 2))]).astype(np.float32)
    maps = uncertainty_maps(original, views)
    assert set(maps) == {"I4_MPC", "I5_Entropy", "I6_PE", "I7_EE", "I8_PV", "I9_MI", "I10_EPKL", "I11_VD"}
    assert np.allclose(maps["I6_PE"], 1.0)
    mean, count = aggregate(maps["I6_PE"], np.ones((2, 2), bool), np.zeros((2, 2), bool))
    assert np.isnan(mean) and count == 0
