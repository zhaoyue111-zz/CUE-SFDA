from __future__ import annotations

import numpy as np
import pytest
import torch

from cue_sfda.high_selection.base import should_select_high
from cue_sfda.high_selection.cross_case_validation import decide as decide_cuv
from cue_sfda.high_selection.gradient_agreement import decide as decide_ga
from cue_sfda.high_selection.semantic_subspace import (
    fit_subspace,
    relative_subspace_error,
    robust_threshold,
)
from cue_sfda.high_selection.semantic_update_validation import decide as decide_suv
from cue_sfda.high_selection.statistics import flatten_gradients, one_sided_delta_test


def test_ga_accepts_aligned_and_rejects_opposed_gradients():
    candidate = torch.tensor([1.0, 0.0])
    assert decide_ga(candidate, [torch.tensor([2.0, 0.0])])["accepted"]
    assert not decide_ga(candidate, [torch.tensor([-2.0, 0.0])])["accepted"]
    assert not decide_ga(torch.zeros(2), [torch.ones(2)])["accepted"]


def test_cuv_delta_estimate_and_one_sided_test():
    candidate = torch.tensor([1.0, 0.0])
    refs = [torch.tensor([2.0, 0.0]), torch.tensor([3.0, 0.0]), torch.tensor([4.0, 0.0])]
    details = one_sided_delta_test(candidate, refs, learning_rate=0.1, significance_level=0.05)
    assert details["mean"] == pytest.approx(-0.3)
    assert details["std"] == pytest.approx(0.1)
    assert details["upper_bound"] < 0.0
    assert decide_cuv(candidate, refs, 0.1, 0.05)["accepted"]
    assert not one_sided_delta_test(candidate, [torch.ones(2)], 0.1, 0.05)["accepted"]


def test_css_svd_projection_and_leave_one_case_out():
    cases = {
        "a": np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32),
        "b": np.array([[1.0, 0.1, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32),
        "c": np.array([[0.9, 0.2, 0.0], [0.7, 0.3, 0.0]], dtype=np.float32),
    }
    errors = []
    for case_id, feature in cases.items():
        basis, rank, energy = fit_subspace({key: value for key, value in cases.items() if key != case_id}, 0.9)
        assert rank >= 1 and energy >= 0.9
        errors.append(relative_subspace_error(feature, basis))
    assert np.isfinite(robust_threshold(errors))
    with pytest.raises(ValueError):
        fit_subspace({"a": cases["a"]}, 0.9)


def test_suv_requires_both_conditions():
    assert decide_suv(True, True)["accepted"]
    assert not decide_suv(True, False)["accepted"]
    assert not decide_suv(False, True)["accepted"]


def test_rejected_high_update_leaves_parameters_and_optimizer_state_unchanged():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    before_parameter = parameter.detach().clone()
    before_state = {key: value.clone() if torch.is_tensor(value) else value for key, value in optimizer.state[parameter].items()}
    parameter.grad = torch.tensor([10.0])
    # A rejected selection uses the conditional AMP finalizer, which must not step.
    from cue_sfda.high_selection.amp import conditional_scaler_step
    scaler = _FakeGradScaler(optimizer)
    before_scaler = scaler.state_dict().copy()
    assert not conditional_scaler_step(
        scaler, optimizer, [parameter], accepted=False, effective=True,
        overflow=False, max_norm=1.0, scale_before=scaler.get_scale(),
    )
    assert torch.equal(parameter.detach(), before_parameter)
    assert optimizer.state[parameter].keys() == before_state.keys()
    for key, value in before_state.items():
        if torch.is_tensor(value):
            assert torch.equal(optimizer.state[parameter][key], value)
        else:
            assert optimizer.state[parameter][key] == value
    assert scaler.state_dict() == before_scaler


def test_low_branch_and_none_mode_bypass_high_selection():
    assert not should_select_high(False, "gradient_agreement")
    assert not should_select_high(True, "none")
    assert should_select_high(True, "semantic_subspace")


def test_invalid_gradients_and_features_fail_closed():
    assert flatten_gradients((None, torch.zeros(2))) is None
    assert flatten_gradients((torch.tensor([float("nan")]),)) is None
    assert not decide_ga(torch.ones(2), [torch.tensor([float("inf"), 0.0])])["accepted"]
    with pytest.raises(ValueError):
        relative_subspace_error(np.empty((0, 3), dtype=np.float32), np.eye(3, 1, dtype=np.float32))


class _FakeGradScaler:
    def __init__(self, optimizer, found_inf=0.0, scale=8.0):
        self.optimizer = optimizer
        self.enabled = True
        self.scale_value = float(scale)
        self.growth = 4
        self.updates = 0
        self.scale_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self._per_optimizer_states = {
            id(optimizer): {"found_inf_per_device": {"cpu": torch.tensor(found_inf)}}
        }

    def is_enabled(self):
        return self.enabled

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        self.scale_calls += 1
        return loss * self.scale_value

    def unscale_(self, optimizer):
        self.unscale_calls += 1
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)
        found_inf = any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for group in optimizer.param_groups for parameter in group["params"]
        )
        self._per_optimizer_states[id(optimizer)] = {
            "found_inf_per_device": {"cpu": torch.tensor(float(found_inf))}
        }

    def step(self, optimizer):
        self.step_calls += 1
        found_inf = self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]["cpu"]
        if float(found_inf.item()) == 0.0:
            optimizer.step()

    def update(self, new_scale=None):
        self.updates += 1
        if new_scale is not None:
            self.scale_value = float(new_scale)
        else:
            found_inf = self._per_optimizer_states[id(self.optimizer)]["found_inf_per_device"]["cpu"]
            if float(found_inf.item()) != 0.0:
                self.scale_value /= 2.0
            else:
                self.growth += 1
        self._per_optimizer_states.clear()

    def state_dict(self):
        return {"scale": self.scale_value, "growth": self.growth}


def _amp_fixture(found_inf=0.0):
    from cue_sfda.high_selection.amp import conditional_scaler_step

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([2.0])
    scaler = _FakeGradScaler(optimizer, found_inf=found_inf)
    return conditional_scaler_step, parameter, optimizer, scaler


def test_amp_high_accept_uses_scaler_step_and_counts_real_update():
    step, parameter, optimizer, scaler = _amp_fixture()
    executed = step(
        scaler, optimizer, [parameter], accepted=True, effective=True,
        overflow=False, max_norm=1.0, scale_before=scaler.get_scale(),
    )
    assert executed
    assert not torch.equal(parameter.detach(), torch.tensor([1.0]))
    assert scaler.updates == 1


def test_amp_high_reject_preserves_model_optimizer_and_scaler_state():
    step, parameter, optimizer, scaler = _amp_fixture()
    before_parameter = parameter.detach().clone()
    before_optimizer = {key: value.clone() if torch.is_tensor(value) else value for key, value in optimizer.state[parameter].items()}
    before_scaler = scaler.state_dict().copy()
    assert not step(
        scaler, optimizer, [parameter], accepted=False, effective=True,
        overflow=False, max_norm=1.0, scale_before=scaler.get_scale(),
    )
    assert torch.equal(parameter.detach(), before_parameter)
    assert optimizer.state[parameter].keys() == before_optimizer.keys()
    assert scaler.state_dict() == before_scaler
    assert scaler.updates == 1  # same-scale update only clears transient UNSCALED state


def test_amp_overflow_skips_optimizer_and_updates_scaler_backoff():
    step, parameter, optimizer, scaler = _amp_fixture(found_inf=1.0)
    before_parameter = parameter.detach().clone()
    assert not step(
        scaler, optimizer, [parameter], accepted=True, effective=True,
        overflow=True, max_norm=1.0, scale_before=scaler.get_scale(),
    )
    assert torch.equal(parameter.detach(), before_parameter)
    assert scaler.get_scale() == 4.0
    assert not optimizer.state[parameter]


def test_gradient_vectors_use_same_parameter_order_for_ga_and_cuv():
    left = flatten_gradients((torch.tensor([1.0, 2.0]), torch.tensor([3.0])))
    right = flatten_gradients((torch.tensor([1.0, 2.0]), torch.tensor([3.0])))
    assert torch.equal(left, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(left, right)
    assert decide_ga(left, [right])["accepted"]
    assert one_sided_delta_test(left, [right, right], 0.1, 0.05)["mean"] < 0


def test_semantic_cache_is_candidate_excluded_precomputed_and_roundtrips():
    from cue_sfda.high_selection.reference_bank import ReferenceBank
    from cue_sfda.high_selection.semantic_subspace import build_semantic_cache

    rng = np.random.default_rng(12)
    features = {
        f"case_{index}": (rng.normal(size=(6, 4)) + np.array([2.0, 0.0, 0.0, 0.0])).astype(np.float32)
        for index in range(8)
    }
    evidence = build_semantic_cache(features, energy=0.9, multiplier=3.0)
    expected = relative_subspace_error(features["case_0"], evidence["case_0"]["basis"])
    bank = ReferenceBank()
    bank.replace(gradients={}, features=features, reference_case_ids=list(features), semantic_evidence=evidence,
                 generated_iteration=20, model_fingerprint="snapshot-20")
    assert bank.semantic_for("case_0")["reference_case_count"] == 7
    assert "case_0" not in bank.feature_references("case_0")
    expected_basis, _, _ = fit_subspace(
        {key: value for key, value in features.items() if key != "case_0"}, 0.9
    )
    cached_basis = bank.semantic_for("case_0")["basis"].numpy()
    assert np.allclose(cached_basis @ cached_basis.T, expected_basis @ expected_basis.T)
    restored = ReferenceBank()
    restored.load_state_dict(bank.state_dict())
    assert restored.generated_iteration == 20
    assert restored.model_fingerprint == "snapshot-20"
    assert relative_subspace_error(features["case_0"], restored.semantic_for("case_0")["basis"].numpy()) == pytest.approx(expected)


def test_semantic_cache_refresh_recomputes_threshold_and_version():
    from cue_sfda.high_selection.reference_bank import ReferenceBank
    from cue_sfda.high_selection.semantic_subspace import build_semantic_cache

    rng = np.random.default_rng(4)
    first = {f"r{i}": rng.normal(size=(5, 3)).astype(np.float32) for i in range(6)}
    second = {key: value + np.array([2.0, 0.0, 0.0], dtype=np.float32) for key, value in first.items()}
    bank = ReferenceBank()
    bank.replace(gradients={}, features=first, reference_case_ids=list(first),
                 semantic_evidence=build_semantic_cache(first), generated_iteration=3,
                 model_fingerprint="a")
    old_threshold = bank.semantic_for("external")["threshold"]
    bank.replace(gradients={}, features=second, reference_case_ids=list(second),
                 semantic_evidence=build_semantic_cache(second), generated_iteration=9,
                 model_fingerprint="b")
    assert bank.cache_version == 2
    assert bank.generated_iteration == 9
    expected = build_semantic_cache(second)["__external_candidate__"]["threshold"]
    assert bank.semantic_for("external")["threshold"] == pytest.approx(expected)
    assert np.isfinite(old_threshold)


class _FeatureModel(torch.nn.Module):
    def __init__(self, logits, features):
        super().__init__()
        self.logits = logits
        self.features = features

    def forward(self, image, embedding, return_features=False):
        return self.logits, self.features


def test_feature_extractor_indexes_real_five_dimensional_tensors():
    from contextlib import nullcontext
    from cue_sfda.high_selection.feature_extractor import foreground_d5_tokens

    logits = torch.full((1, 1, 2, 2, 2), -10.0)
    logits[0, 0, 1, 0, 1] = 10.0
    features = torch.zeros((1, 3, 2, 2, 2))
    features[0, :, 1, 0, 1] = torch.tensor([2.0, 3.0, 4.0])
    tokens = foreground_d5_tokens(
        _FeatureModel(logits, features), torch.zeros((1, 1, 2, 2, 2)),
        torch.zeros((1, 1, 1)), 0.5, 16, 7, nullcontext,
    )
    assert tokens.shape == (1, 3)
    assert np.array_equal(tokens[0], np.array([2.0, 3.0, 4.0], dtype=np.float32))


def test_feature_extractor_nearest_aligns_different_logits_spatial_shape():
    from contextlib import nullcontext
    from cue_sfda.high_selection.feature_extractor import foreground_d5_tokens

    logits = torch.full((1, 1, 4, 4, 4), -10.0)
    logits[0, 0, 0, 0, 0] = 10.0
    features = torch.arange(16.0).reshape(1, 2, 2, 2, 2)
    tokens = foreground_d5_tokens(
        _FeatureModel(logits, features), torch.zeros((1, 1, 4, 4, 4)),
        torch.zeros((1, 1, 1)), 0.5, 16, 2, nullcontext,
    )
    assert tokens.shape == (1, 2)
    assert np.array_equal(tokens[0], np.array([0.0, 8.0], dtype=np.float32))


def test_feature_extractor_returns_empty_for_no_foreground():
    from contextlib import nullcontext
    from cue_sfda.high_selection.feature_extractor import foreground_d5_tokens

    tokens = foreground_d5_tokens(
        _FeatureModel(torch.full((1, 1, 2, 2, 2), -10.0), torch.ones((1, 2, 2, 2, 2))),
        torch.zeros((1, 1, 2, 2, 2)), torch.zeros((1, 1, 1)),
        0.5, 8, 3, nullcontext,
    )
    assert tokens.shape == (0, 2)


def test_feature_extractor_rejects_nonfinite_and_invalid_batch_shapes():
    from contextlib import nullcontext
    from cue_sfda.high_selection.feature_extractor import foreground_d5_tokens

    logits = torch.zeros((1, 1, 2, 2, 2))
    features = torch.zeros((1, 2, 2, 2, 2))
    features[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        foreground_d5_tokens(
            _FeatureModel(logits, features), torch.zeros((1, 1, 2, 2, 2)),
            torch.zeros((1, 1, 1)), 0.5, 16, 1, nullcontext,
        )
    with pytest.raises(ValueError, match="batch"):
        foreground_d5_tokens(
            _FeatureModel(torch.zeros((2, 1, 2, 2, 2)), torch.zeros((2, 2, 2, 2, 2))),
            torch.zeros((1, 1, 2, 2, 2)), torch.zeros((1, 1, 1)),
            0.5, 16, 1, nullcontext,
        )


def _fake_trainer_for_high_step(accepted, *, loss_scale=8.0, loss_factor=1.0):
    from types import SimpleNamespace
    from cue_sfda.engine.trainer import CUETrainer

    trainer = object.__new__(CUETrainer)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scaler = _FakeGradScaler(optimizer, scale=loss_scale)
    case = SimpleNamespace(name="high_case")
    trainer.partition = SimpleNamespace(
        is_high_confidence=lambda name: True,
        scores={"high_case": 1.0},
        update=lambda *args: None,
    )
    trainer.store = SimpleNamespace(
        sliding_window=lambda record, step: (torch.zeros((1, 1, 1, 1)), [(slice(0, 1),) * 3])
    )
    trainer._patch = lambda volume, slicer: torch.ones((1, 1, 1, 1, 1))
    trainer._high_confidence_loss = lambda image: (
        parameter.square() * float(loss_factor),
        {"raw_foreground_voxels": 1, "refined_foreground_voxels": 1,
         "voxels": 1, "effective": True},
    )
    trainer._low_confidence_loss = lambda *args: (_ for _ in ()).throw(AssertionError("Low path touched"))
    trainer._decide_high_update = lambda record, candidate, **kwargs: {
        "accepted": bool(accepted), "reason": "fake_decision",
    }
    trainer._write_high_selection_row = lambda *args, **kwargs: None
    trainer.config = {"train": {"max_grad_norm": 1.0}, "method": {}}
    trainer.tile_step_size = 1.0
    trainer.optimizer = optimizer
    trainer.runtime = SimpleNamespace(trainable_parameters=[parameter])
    trainer.scaler = scaler
    trainer.high_selection_mode = "gradient_agreement"
    trainer.high_selection_pending_row = None
    trainer.high_selection_attempts = trainer.high_selection_accepts = trainer.high_selection_rejects = 0
    trainer.high_selection_cache_iteration = None
    trainer.high_selection_rng = np.random.default_rng(0)
    trainer.dynamic_hard_enabled = False
    trainer.optimizer_step_count = 0
    trainer.branch_sample_counts = {"high": 0, "low": 0}
    trainer.last_training_iteration = 0
    return trainer, case, parameter, optimizer, scaler


def test_train_step_amp_high_accept_unscales_once_then_uses_scaler_step():
    from cue_sfda.engine.trainer import CUETrainer

    trainer, case, parameter, optimizer, scaler = _fake_trainer_for_high_step(True)
    observed = {}
    def accept(record, candidate, **kwargs):
        observed["candidate"] = candidate.clone()
        return {"accepted": True, "reason": "accepted"}
    trainer._decide_high_update = accept
    CUETrainer._train_step(trainer, case, case)
    assert torch.allclose(observed["candidate"], torch.tensor([2.0]))
    assert scaler.scale_calls == 1 and scaler.unscale_calls == 1
    assert scaler.step_calls == 1
    assert trainer.optimizer_step_count == 1


def test_train_step_amp_high_rejection_preserves_parameter_optimizer_and_scaler():
    from cue_sfda.engine.trainer import CUETrainer

    trainer, case, parameter, optimizer, scaler = _fake_trainer_for_high_step(False)
    before_parameter = parameter.detach().clone()
    before_optimizer = optimizer.state_dict()
    before_scaler = scaler.state_dict().copy()
    CUETrainer._train_step(trainer, case, case)
    assert torch.equal(parameter.detach(), before_parameter)
    assert optimizer.state_dict() == before_optimizer
    assert scaler.state_dict() == before_scaler
    assert scaler.unscale_calls == 1 and scaler.step_calls == 0
    assert trainer.optimizer_step_count == 0


def test_train_step_amp_overflow_skips_optimizer_and_step_counter():
    from cue_sfda.engine.trainer import CUETrainer

    trainer, case, parameter, optimizer, scaler = _fake_trainer_for_high_step(
        True, loss_scale=1e20, loss_factor=1e20
    )
    before_parameter = parameter.detach().clone()
    CUETrainer._train_step(trainer, case, case)
    assert torch.equal(parameter.detach(), before_parameter)
    assert not optimizer.state[parameter]
    assert scaler.step_calls == 1  # GradScaler sees recorded Inf and skips optimizer.step.
    assert scaler.get_scale() == 5e19
    assert trainer.optimizer_step_count == 0


def test_checkpoint_restore_preserves_older_but_compatible_reference_snapshot():
    import random
    from cue_sfda.engine.trainer import CUETrainer
    from cue_sfda.high_selection.reference_bank import ReferenceBank
    from cue_sfda.high_selection.semantic_subspace import build_semantic_cache

    rng = np.random.default_rng(21)
    features = {f"id{i}": rng.normal(size=(5, 3)).astype(np.float32) for i in range(6)}
    bank = ReferenceBank()
    bank.replace(
        gradients={"id0": torch.tensor([1.0, 2.0])},
        features=features,
        reference_case_ids=list(features),
        semantic_evidence=build_semantic_cache(features),
        generated_iteration=12,
        model_fingerprint="student-at-cache-12",
    )
    trainer = object.__new__(CUETrainer)
    trainer.high_selection_mode = "semantic_update_validation"
    trainer.high_selection_cache_iteration = None
    trainer.high_reference_bank = ReferenceBank()
    trainer.high_selection_refs = trainer.high_reference_bank.gradients
    trainer.high_selection_features = trainer.high_reference_bank.features
    trainer.high_selection_rng = random.Random(0)
    trainer.high_selection_attempts = trainer.high_selection_accepts = trainer.high_selection_rejects = 0
    trainer.optimizer_step_count = 0
    trainer.last_training_iteration = 20
    trainer._high_selection_config_fingerprint = lambda: "config-fingerprint"
    trainer._lora_fingerprint = lambda: "student-at-checkpoint-20"
    class _Logger:
        def warning(self, *args, **kwargs):
            raise AssertionError("valid checkpoint must not use recovery path")
    trainer.logger = _Logger()
    state = {
        "high_selection_state": {
            "schema_version": 1,
            "mode": "semantic_update_validation",
            "config_fingerprint": "config-fingerprint",
            "checkpoint_student_fingerprint": "student-at-checkpoint-20",
            "cache_iteration": 12,
            "cache_version": bank.cache_version,
            "cache_model_fingerprint": "student-at-cache-12",
            "reference_bank": bank.state_dict(),
            "rng_state": trainer.high_selection_rng.getstate(),
            "attempts": 8,
            "accepts": 3,
            "rejects": 5,
            "optimizer_step_count": 17,
        }
    }
    CUETrainer._restore_high_selection_state(trainer, state)
    assert trainer.high_selection_cache_iteration == 12
    assert trainer.high_reference_bank.model_fingerprint == "student-at-cache-12"
    assert trainer.high_reference_bank.model_fingerprint != "student-at-checkpoint-20"
    assert trainer.high_selection_attempts == 8
    assert trainer.optimizer_step_count == 17
    assert trainer.high_reference_bank.semantic_for("id0")["rank"] > 0
