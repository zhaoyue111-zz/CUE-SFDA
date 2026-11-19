from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch import nn

from cue_sfda.data import CaseRecord
from cue_sfda.high_selection.feature_extractor import foreground_d5_tokens
from cue_sfda.high_selection.reference_bank import ReferenceBank
from cue_sfda.high_selection.semantic_subspace import (
    build_semantic_cache,
    fit_subspace,
    relative_subspace_error,
)
from cue_sfda.method.uncertainty import ConfidencePartition


class _FeatureModel(nn.Module):
    def __init__(self, logits, features):
        super().__init__()
        self.logits = logits
        self.features = features

    def forward(self, image, embedding, return_features=False):
        return self.logits, self.features


def _no_autocast():
    return nullcontext()


def test_foreground_feature_indexing_preserves_3d_batch_and_channel_axes():
    logits = torch.full((1, 1, 2, 2, 2), -10.0)
    logits[0, 0, 1, 0, 1] = 10.0
    features = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 2, 2)
    model = _FeatureModel(logits, features)

    tokens = foreground_d5_tokens(
        model,
        torch.zeros((1, 1, 2, 2, 2)),
        torch.zeros((1, 1, 2)),
        probability_threshold=0.5,
        max_tokens=8,
        seed=3,
        autocast=_no_autocast,
    )

    expected = features[0, :, 1, 0, 1].numpy()[None, :]
    np.testing.assert_array_equal(tokens, expected)


def test_foreground_feature_mask_is_aligned_when_logits_and_features_differ():
    logits = torch.full((1, 1, 4, 4, 4), -10.0)
    logits[0, 0, 0, 0, 0] = 10.0
    features = torch.tensor([[[[[7.0, 9.0]]], [[[2.0, 4.0]]]]])
    model = _FeatureModel(logits, features)

    tokens = foreground_d5_tokens(
        model,
        torch.zeros((1, 1, 4, 4, 4)),
        torch.zeros((1, 1, 2)),
        probability_threshold=0.5,
        max_tokens=8,
        seed=1,
        autocast=_no_autocast,
    )

    np.testing.assert_array_equal(tokens, np.array([[7.0, 2.0]], dtype=np.float32))


def test_foreground_feature_extraction_fails_closed_on_nonfinite_values():
    logits = torch.zeros((1, 1, 2, 2, 2))
    features = torch.ones((1, 2, 2, 2, 2))
    features[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        foreground_d5_tokens(
            _FeatureModel(logits, features),
            torch.zeros((1, 1, 2, 2, 2)),
            torch.zeros((1, 1, 2)),
            probability_threshold=0.5,
            max_tokens=8,
            seed=1,
            autocast=_no_autocast,
        )


def test_css_cache_precomputes_candidate_excluded_bases_and_thresholds():
    rng = np.random.default_rng(17)
    features = {
        f"case_{index}": (
            rng.normal(size=(12, 5)) * 0.05
            + np.eye(5, dtype=np.float32)[index % 5]
        ).astype(np.float32)
        for index in range(8)
    }
    evidence = build_semantic_cache(features, energy=0.9, multiplier=3.0)
    bank = ReferenceBank()
    bank.replace(
        gradients={},
        features=features,
        reference_case_ids=list(features),
        semantic_evidence=evidence,
        generated_iteration=40,
        model_fingerprint="snapshot-40",
    )

    assert "case_0" not in bank.feature_references("case_0")
    candidate_evidence = bank.semantic_for("case_0")
    expected_basis, expected_rank, expected_energy = fit_subspace(
        {name: value for name, value in features.items() if name != "case_0"}, 0.9
    )
    assert candidate_evidence["rank"] == expected_rank
    assert candidate_evidence["spectral_energy"] == pytest.approx(expected_energy)
    assert candidate_evidence["basis"].device.type == "cpu"
    assert np.allclose(
        candidate_evidence["basis"].numpy() @ candidate_evidence["basis"].numpy().T,
        expected_basis @ expected_basis.T,
    )
    assert relative_subspace_error(features["case_0"], candidate_evidence["basis"].numpy()) >= 0.0
    assert candidate_evidence["reference_case_count"] == 7


def test_css_reference_refresh_recomputes_robust_threshold():
    rng = np.random.default_rng(23)
    first = {
        f"case_{index}": rng.normal(loc=0.0, scale=0.05, size=(10, 4)).astype(np.float32)
        for index in range(8)
    }
    second = {name: value.copy() for name, value in first.items()}
    second["case_7"] += np.array([4.0, -3.0, 2.0, 1.0], dtype=np.float32)

    first_evidence = build_semantic_cache(first, energy=0.9, multiplier=3.0)
    refreshed_evidence = build_semantic_cache(second, energy=0.9, multiplier=3.0)

    assert refreshed_evidence["__external_candidate__"]["threshold"] != pytest.approx(
        first_evidence["__external_candidate__"]["threshold"]
    )


def test_css_uses_frozen_g_dynamic_hard_high_low_assignments(tmp_path):
    import csv
    from types import SimpleNamespace

    from cue_sfda.engine.trainer import CUETrainer

    route_path = tmp_path / "routing.csv"
    with route_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "assigned_branch"))
        writer.writeheader()
        writer.writerows([
            {"case_id": "high_case.nii.gz", "assigned_branch": "high"},
            {"case_id": "low_case.nii.gz", "assigned_branch": "low"},
        ])
    trainer = CUETrainer.__new__(CUETrainer)
    trainer.config = {"method": {"routing_assignments_csv": str(route_path)}}
    trainer.train_cases = [
        CaseRecord("high_case.nii.gz", tmp_path / "high_case.nii.gz"),
        CaseRecord("low_case.nii.gz", tmp_path / "low_case.nii.gz"),
    ]
    trainer.partition = ConfidencePartition(quality_threshold=0.5, momentum=0.9)
    trainer.output_dir = tmp_path / "CSS"
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    trainer.skipped_case_names = set()
    trainer.routing_metadata = {}

    trainer._initialize_routing("joint")

    assert trainer.partition.is_high_confidence("high_case.nii.gz")
    assert not trainer.partition.is_high_confidence("low_case.nii.gz")
    assert trainer.routing_metadata["high"] == 1
    assert trainer.routing_metadata["low"] == 1
    assert (trainer.output_dir / "routing.csv").read_text(encoding="utf-8") == route_path.read_text(encoding="utf-8")
