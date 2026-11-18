import random
from pathlib import Path

import pytest

from cue_sfda.data import CaseRecord
from cue_sfda.engine.high_quality_warmup import shuffled_epoch_cases
from cue_sfda.engine.low_only_dynamic_hard import _summarize_metrics


def test_low_only_epoch_visits_every_low_case_once_and_shuffles():
    low = [CaseRecord(f"low-{index}", Path(f"low-{index}")) for index in range(42)]
    rng = random.Random(42)
    first = shuffled_epoch_cases(low, rng)
    second = shuffled_epoch_cases(low, rng)
    assert len(first) == len(second) == 42
    assert {case.name for case in first} == {case.name for case in low}
    assert {case.name for case in second} == {case.name for case in low}
    assert [case.name for case in first] != [case.name for case in second]


def test_low_only_evaluation_summarizes_all_groups_and_metrics():
    groups = {"h1": "high", "h2": "high", "l1": "low"}
    baseline = {
        "h1": {"dice": 0.2, "iou": 0.1, "precision": 0.3, "recall": 0.4},
        "h2": {"dice": 0.4, "iou": 0.3, "precision": 0.5, "recall": 0.6},
        "l1": {"dice": 0.1, "iou": 0.05, "precision": 0.2, "recall": 0.25},
    }
    current = {
        "h1": {"dice": 0.3, "iou": 0.2, "precision": 0.4, "recall": 0.5},
        "h2": {"dice": 0.5, "iou": 0.4, "precision": 0.6, "recall": 0.7},
        "l1": {"dice": 0.2, "iou": 0.1, "precision": 0.3, "recall": 0.35},
    }
    summaries, per_case = _summarize_metrics(5, current, groups, baseline)
    by_group = {row["group"]: row for row in summaries}
    assert by_group["all"]["case_count"] == 3
    assert by_group["high"]["mean_dice"] == pytest.approx(0.4)
    assert by_group["low"]["delta_recall"] == pytest.approx(0.1)
    assert len(per_case) == 3
    assert per_case[-1]["delta_precision"] == pytest.approx(0.1)


def test_low_only_evaluation_rejects_missing_or_extra_cases():
    groups = {"high": "high", "low": "low"}
    with pytest.raises(ValueError, match="differ from saved routing"):
        _summarize_metrics(
            0,
            {"low": {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}},
            groups,
            None,
        )
