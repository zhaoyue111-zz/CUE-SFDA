from pathlib import Path
import random

import pytest
import torch
from torch import nn

from cue_sfda.data import CaseRecord
from cue_sfda.engine.checkpoint import load_checkpoint, save_checkpoint
from cue_sfda.engine.high_quality_warmup import (
    load_and_validate_routing,
    should_replace_best,
    shuffled_epoch_cases,
    summarize_epoch_metrics,
)
from cue_sfda.modeling.lora import inject_decoder_qkv_lora


def cases(*names):
    return [CaseRecord(name=name, image_path=Path(name)) for name in names]


def write_route(path, rows):
    path.write_text(
        "case_id,response_state,assigned_branch\n"
        + "".join(f"{case_id},{state},{branch}\n" for case_id, state, branch in rows),
        encoding="utf-8",
    )


def test_route_requires_exact_disjoint_partition_and_empty_low(tmp_path):
    path = tmp_path / "routing.csv"
    write_route(path, [("a", "response-present", "high"), ("b", "No-response", "low")])
    groups, _ = load_and_validate_routing(path, cases("a", "b"))
    assert groups == {"a": "high", "b": "low"}

    write_route(path, [("a", "No-response", "high"), ("b", "response-present", "low")])
    with pytest.raises(ValueError, match="No-response"):
        load_and_validate_routing(path, cases("a", "b"))
    write_route(path, [("a", "response-present", "high")])
    with pytest.raises(ValueError, match="exactly match"):
        load_and_validate_routing(path, cases("a", "b"))


def test_epoch_order_visits_every_high_case_once_and_is_seeded():
    high = cases(*(f"case-{index}" for index in range(59)))
    first = shuffled_epoch_cases(high, random.Random(42))
    second = shuffled_epoch_cases(high, random.Random(42))
    next_epoch = shuffled_epoch_cases(high, random.Random(42))
    assert len(first) == 59
    assert {case.name for case in first} == {case.name for case in high}
    assert [case.name for case in first] == [case.name for case in second]
    # The same RNG instance advances between epochs, so each epoch gets a new order.
    rng = random.Random(42)
    epoch_one = shuffled_epoch_cases(high, rng)
    epoch_two = shuffled_epoch_cases(high, rng)
    assert [case.name for case in epoch_one] != [case.name for case in epoch_two]
    assert len(next_epoch) == 59


def test_evaluation_summary_covers_groups_and_deltas():
    groups = {"a": "high", "b": "high", "c": "low"}
    baseline, _ = summarize_epoch_metrics(0, {"a": 0.1, "b": 0.3, "c": 0.5}, groups, None)
    current, _ = summarize_epoch_metrics(5, {"a": 0.2, "b": 0.5, "c": 0.4}, groups, baseline)
    assert current["high_mean_dice"] == pytest.approx(0.35)
    assert current["low_mean_dice"] == pytest.approx(0.4)
    assert current["delta_high"] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="differ from saved routing"):
        summarize_epoch_metrics(10, {"a": 0.1}, groups, baseline)


def test_best_selection_only_replaces_for_better_score_or_earlier_tie():
    assert should_replace_best(0.4, 5, None, None)
    assert should_replace_best(0.4, 5, 0.4, 10)
    assert not should_replace_best(0.4, 10, 0.4, 5)
    assert not should_replace_best(0.39, 15, 0.4, 10)


def test_warmup_checkpoint_round_trips_through_existing_loader(tmp_path):
    model = nn.Module()
    model.multihead_attn = nn.MultiheadAttention(8, 2)
    inject_decoder_qkv_lora(model, rank=2, alpha=2.0, dropout=0.0)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    with torch.no_grad():
        model.multihead_attn.lora_q_B.fill_(0.125)
    path = tmp_path / "best_high.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        iteration=5,
        config={"experiment": {"seed": 42}},
        confidence_scores={"high-case": 1.0, "low-case": 0.0},
        confidence_threshold=0.5,
        best_dice=0.4,
        best_iteration=5,
        trainer_state={"warmup_epoch": 5, "train_group": "high_only"},
    )
    with torch.no_grad():
        model.multihead_attn.lora_q_B.zero_()
    restored = load_checkpoint(path, model, optimizer)
    assert torch.all(model.multihead_attn.lora_q_B == 0.125)
    assert restored["trainer_state"]["warmup_epoch"] == 5
    assert restored["trainer_state"]["train_group"] == "high_only"
