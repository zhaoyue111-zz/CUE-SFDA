import csv
import json

from cue_sfda.method.routing import build_routing_assignments


def _inputs(tmp_path, empty_uncertainty_first=False):
    uncertainty = tmp_path / "uncertainty.csv"
    with uncertainty.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case_id", "image_uncertainty", "prompt_uncertainty"),
        )
        writer.writeheader()
        for index in range(5):
            image_uncertainty = index / 10
            prompt_uncertainty = (4 - index) / 10
            if empty_uncertainty_first and index == 4:
                image_uncertainty = 0.0
                prompt_uncertainty = 0.0
            writer.writerow({
                "case_id": f"case{index}",
                "image_uncertainty": image_uncertainty,
                "prompt_uncertainty": prompt_uncertainty,
            })
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({
        "threshold": 0.5,
        "scores": {f"case{i}.nii.gz": float(i < 3) for i in range(5)},
        "foreground_counts": {f"case{i}.nii.gz": 0 if i == 4 else i + 1 for i in range(5)},
    }))
    return uncertainty, partition


def test_image_routing_preserves_original_high_low_counts(tmp_path):
    uncertainty, partition = _inputs(tmp_path)
    rows, _, metadata = build_routing_assignments(
        [f"case{i}.nii.gz" for i in range(5)], "image", uncertainty, partition
    )
    assert metadata["high"] == 3
    assert metadata["low"] == 2
    assert metadata["skip"] == 0
    assert [row.case_id for row in rows if row.assigned_branch == "high"] == [
        "case0.nii.gz", "case1.nii.gz", "case2.nii.gz"
    ]


def test_joint_gate_skips_only_empty_responses(tmp_path):
    uncertainty, partition = _inputs(tmp_path)
    rows, _, metadata = build_routing_assignments(
        [f"case{i}.nii.gz" for i in range(5)], "joint_gate", uncertainty, partition
    )
    skipped = [row for row in rows if row.assigned_branch == "skip"]
    assert [row.case_id for row in skipped] == ["case4.nii.gz"]
    assert skipped[0].response_state == "No-response"
    assert metadata["no_response_all_raw_predictions_empty"] is True


def test_joint_empty_to_low_keeps_empty_predictions_low_and_selects_joint_high(tmp_path):
    uncertainty, partition = _inputs(tmp_path, empty_uncertainty_first=True)
    rows, scores, metadata = build_routing_assignments(
        [f"case{i}.nii.gz" for i in range(5)],
        "joint_empty_to_low",
        uncertainty,
        partition,
        target_high_count=3,
    )
    by_case = {row.case_id: row for row in rows}

    assert metadata["high"] == 3
    assert metadata["low"] == 2
    assert metadata["skip"] == 0
    assert metadata["empty_predictions_forced_low"] is True
    assert by_case["case4.nii.gz"].response_state == "No-response"
    assert by_case["case4.nii.gz"].assigned_branch == "low"
    assert [row.case_id for row in rows if row.assigned_branch == "high"] == [
        "case0.nii.gz", "case1.nii.gz", "case2.nii.gz"
    ]
    assert scores["case4.nii.gz"] == 0.0
