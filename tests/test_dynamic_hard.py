import numpy as np
import torch
import pytest
from types import SimpleNamespace
from torch import nn

from cue_sfda.data import CaseRecord
from cue_sfda.method.dynamic_hard import (
    DynamicHardCache,
    DynamicHardCacheBuilder,
    dynamic_hard_consensus_loss,
    sample_coverage_index,
)
from cue_sfda.modeling.lora import LoRALinear
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
        "cache_version": 0,
        "teacher_iteration": None,
    }
    initial = np.zeros((1, 2, 2, 2), dtype=np.float32)
    refreshed = np.ones((1, 2, 2, 2), dtype=np.float32)
    confidence = np.full((1, 2, 2, 2), 0.5, dtype=np.float32)

    cache.save_case(case, 0, initial, confidence, {**geometry, "teacher_type": "frozen_source"})
    cache.save_case(
        case,
        1,
        refreshed,
        confidence,
        {
            **geometry,
            "cache_version": 1,
            "teacher_type": "student_snapshot",
            "teacher_iteration": 5,
        },
    )
    common = {
        "schema_version": 1,
        "num_cases": 1,
        "case_ids": [case.name],
        "num_views": 10,
        "prompts": ["liver cancer"],
        "standard_prompt": "liver cancer",
        "image_temperature": 0.15,
        "prompt_temperature": 0.15,
        "refresh_tile_step_size": 1.0,
        "source_checkpoint": None,
    }
    cache.write_version_metadata(
        0,
        {**common, "cache_version": 0, "teacher_type": "frozen_source"},
    )
    cache.write_version_metadata(
        1,
        {
            **common,
            "cache_version": 1,
            "teacher_type": "student_snapshot",
            "teacher_iteration": 5,
        },
    )
    cache.activate_version(0)

    np.testing.assert_array_equal(
        cache.load_case(case, 0)["joint_probability"], initial
    )
    np.testing.assert_array_equal(
        cache.load_case(case, 1)["joint_probability"], refreshed
    )
    assert cache.case_path(case.name, 0) != cache.case_path(case.name, 1)
    assert cache.metadata()["active_version"] == 0
    assert cache.metadata(0)["teacher_type"] == "frozen_source"
    assert cache.metadata(1)["teacher_type"] == "student_snapshot"


def test_coverage_sampling_prefers_unseen_cases_in_both_passes():
    names = [f"high_{index}" for index in range(6)] + [
        f"low_{index}" for index in range(4)
    ]
    high, low = list(range(6)), list(range(6, 10))
    generator = torch.Generator().manual_seed(42)
    seen = set()

    for _cycle in range(2):
        pass_seen = set()
        for _ in range(len(names)):
            index = sample_coverage_index(
                high, low, names, seen, 0.75, generator
            )
            assert names[index] not in seen
            seen.add(names[index])
            pass_seen.add(names[index])
        assert pass_seen == set(names)
        seen.clear()


def test_coverage_sampler_retains_regular_high_low_probability_when_uncovered():
    names = [f"high_{index}" for index in range(12)] + [
        f"low_{index}" for index in range(8)
    ]
    high, low = list(range(12)), list(range(12, 20))
    guided_generator = torch.Generator().manual_seed(82)
    regular_generator = torch.Generator().manual_seed(82)
    guided = [
        sample_coverage_index(high, low, names, set(), 0.75, guided_generator)
        for _ in range(200)
    ]
    regular = []
    for _ in range(200):
        use_high = bool(torch.rand((), generator=regular_generator) < 0.75)
        choices = high if use_high else low
        regular.append(
            choices[
                int(torch.randint(len(choices), (1,), generator=regular_generator))
            ]
        )
    assert guided == regular


def test_dynamic_hard_loss_matches_historical_weighted_bce():
    probability = torch.tensor([[[[[0.2, 0.8]]]]], requires_grad=True)
    target = torch.tensor([[[[[0.0, 1.0]]]]])
    confidence = torch.tensor([[[[[0.5, 0.01]]]]])
    actual = dynamic_hard_consensus_loss(
        probability, target, confidence, confidence_threshold=0.1
    )
    expected_weight = confidence * (confidence >= 0.1).float()
    expected = torch.nn.functional.binary_cross_entropy(
        probability.clamp(1e-6, 1 - 1e-6), target, reduction="none"
    )
    expected = (expected * expected_weight.clamp_min(1e-3)).sum() / expected_weight.clamp_min(1e-3).sum().clamp_min(1.0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("version", "teacher_type"),
    ((0, "frozen_source"), (1, "student_snapshot")),
)
def test_cache_rejects_wrong_prediction_version(tmp_path, version, teacher_type):
    image_path = tmp_path / "case.nii.gz"
    image_path.write_bytes(b"image bytes")
    case = CaseRecord(name=image_path.name, image_path=image_path)
    cache = DynamicHardCache(tmp_path / "cache")
    probability = np.zeros((1, 2, 2, 2), dtype=np.float32)
    confidence = np.ones_like(probability)
    stat = image_path.stat()
    geometry = {
        "case_id": case.name,
        "image_size": int(stat.st_size),
        "image_mtime_ns": int(stat.st_mtime_ns),
        "padded_shape": [2, 2, 2],
        "cache_version": version,
        "teacher_type": "student_snapshot" if version == 0 else "frozen_source",
        "teacher_iteration": None if version == 0 else 123,
    }
    cache.save_case(case, version, probability, confidence, geometry)
    cache.write_version_metadata(
        version,
        {"teacher_type": teacher_type, "cache_version": version},
    )
    with pytest.raises(ValueError, match="version/type"):
        cache.load_case(case, version)


class CacheTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = LoRALinear(nn.Linear(1, 1, bias=False), rank=1, alpha=1.0)
        with torch.no_grad():
            self.projection.base.weight.fill_(0.1)
            self.projection.lora_A.fill_(1.0)
            self.projection.lora_B.fill_(1.0)

    def forward(self, image, _embedding):
        scale = self.projection(torch.ones((1, 1), device=image.device))
        return image * scale.reshape(1, 1, 1, 1, 1)


class CacheTestStore:
    def sliding_window(self, _case, _step):
        return torch.ones((1, 2, 2, 2)), [(slice(0, 2),) * 3]

    def extract_patch(self, volume, slicer):
        return volume[(slice(None), *slicer)]


class CacheTestViews:
    def make_uncertainty_views(self, image):
        return [SimpleNamespace(image=image), SimpleNamespace(image=image)]


class QuietLogger:
    def info(self, *args, **kwargs):
        pass


def test_initial_cache_is_source_and_refresh_cache_uses_lora_student(tmp_path):
    image_path = tmp_path / "case.nii.gz"
    image_path.write_bytes(b"image")
    case = CaseRecord(name=image_path.name, image_path=image_path)
    model = CacheTestModel()
    runtime = SimpleNamespace(model=model, device=torch.device("cpu"))
    config = {
        "method": {"num_views": 2},
        "model": {"amp": False},
        "dynamic_hard": {
            "image_temperature": 0.15,
            "prompt_temperature": 0.15,
            "refresh_tile_step_size": 1.0,
            "standard_prompt": "liver cancer",
            "prompts": ["liver cancer"],
        },
    }
    cache = DynamicHardCache(tmp_path / "cache")
    builder = DynamicHardCacheBuilder(
        config,
        runtime,
        CacheTestStore(),
        CacheTestViews(),
        {"liver cancer": torch.zeros(1)},
        QuietLogger(),
    )
    builder.build([case], cache, 0, "frozen_source")
    source = cache.load_case(case, 0)["joint_probability"]

    builder.disable_lora = False
    builder.build([case], cache, 1, "student_snapshot", teacher_iteration=2)
    student = cache.load_case(case, 1)["joint_probability"]

    assert cache.metadata(0)["teacher_type"] == "frozen_source"
    assert cache.metadata(1)["teacher_type"] == "student_snapshot"
    assert float(student.mean()) > float(source.mean())
