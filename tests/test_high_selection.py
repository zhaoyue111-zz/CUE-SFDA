from __future__ import annotations

import numpy as np
import pytest
import torch

from cue_sfda.high_selection.base import conditional_high_step, should_select_high
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
    assert robust_threshold(errors) >= max(errors)
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
    conditional_high_step(optimizer, accepted=True, effective=True)
    before_parameter = parameter.detach().clone()
    before_state = {key: value.clone() if torch.is_tensor(value) else value for key, value in optimizer.state[parameter].items()}
    parameter.grad = torch.tensor([10.0])
    assert not conditional_high_step(optimizer, accepted=False, effective=True)
    assert torch.equal(parameter.detach(), before_parameter)
    assert optimizer.state[parameter].keys() == before_state.keys()
    for key, value in before_state.items():
        if torch.is_tensor(value):
            assert torch.equal(optimizer.state[parameter][key], value)
        else:
            assert optimizer.state[parameter][key] == value


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
