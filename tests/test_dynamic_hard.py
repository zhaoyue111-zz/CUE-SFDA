import numpy as np

from cue_sfda.data import CaseRecord
from cue_sfda.method.dynamic_hard import (
    DynamicHardCache,
    consensus,
    refresh_ready,
)


def test_consensus_returns_fused_probability_and_full_agreement():
    predictions = np.full((3, 1, 2, 2, 2), 0.7, dtype=np.float32)
    probability, confidence, weights = consensus(predictions, temperature=0.15)

    assert probability.shape == (1, 2, 2, 2)
    assert confidence.shape == probability.shape
    np.testing.assert_allclose(probability, 0.7, atol=1e-6)
    np.testing.assert_allclose(confidence, 1.0, atol=1e-6)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


def test_dynamic_hard_refresh_waits_for_full_coverage_and_runs_once():
    cases = {"a.nii.gz", "b.nii.gz", "c.nii.gz"}

    assert not refresh_ready(cases - {"c.nii.gz"}, cases, 0, 1)
    assert refresh_ready(cases, cases, 0, 1)
    assert not refresh_ready(cases, cases, 1, 1)


def test_dynamic_hard_cache_keeps_versions_separate(tmp_path):
    image_path = tmp_path / "case.nii.gz"
    image_path.write_bytes(b"image bytes")
    case = CaseRecord(name=image_path.name, image_path=image_path)
    cache = DynamicHardCache(tmp_path / "M1_dynamic_hard_cache")
    geometry = {
        "case_id": case.name,
        "image_size": image_path.stat().st_size,
        "image_mtime_ns": image_path.stat().st_mtime_ns,
        "padded_shape": [2, 2, 2],
    }
    initial = np.zeros((1, 2, 2, 2), dtype=np.float32)
    refreshed = np.ones((1, 2, 2, 2), dtype=np.float32)
    confidence = np.full((1, 2, 2, 2), 0.5, dtype=np.float32)

    cache.save_case(case, 0, initial, confidence, geometry)
    cache.save_case(case, 1, refreshed, confidence, geometry)

    np.testing.assert_array_equal(
        cache.load_case(case, 0)["joint_probability"], initial
    )
    np.testing.assert_array_equal(
        cache.load_case(case, 1)["joint_probability"], refreshed
    )
    assert cache.case_path(case.name, 0) != cache.case_path(case.name, 1)
