from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROUTING_FIELDS = (
    "case_id",
    "response_state",
    "image_uncertainty",
    "prompt_uncertainty",
    "joint_uncertainty",
    "assigned_branch",
)


def normalize_case_id(value: str) -> str:
    name = Path(str(value).strip()).name
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


@dataclass(frozen=True)
class RoutingAssignment:
    case_id: str
    response_state: str
    image_uncertainty: float
    prompt_uncertainty: float
    joint_uncertainty: float
    assigned_branch: str

    def as_dict(self) -> dict[str, str | float]:
        return {field: getattr(self, field) for field in ROUTING_FIELDS}


def load_uncertainties(path: str | Path) -> dict[str, dict[str, float]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "image_uncertainty", "prompt_uncertainty"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Uncertainty CSV missing columns: {sorted(missing)}")
        result = {}
        for row in reader:
            case_id = normalize_case_id(row["case_id"])
            if case_id in result:
                raise ValueError(f"Duplicate uncertainty case: {case_id}")
            image = float(row["image_uncertainty"])
            prompt = float(row["prompt_uncertainty"])
            if not (0.0 <= image <= 1.0 and 0.0 <= prompt <= 1.0):
                raise ValueError(f"Invalid uncertainty for {case_id}")
            result[case_id] = {
                "image_uncertainty": image,
                "prompt_uncertainty": prompt,
                "joint_uncertainty": (image + prompt) / 2.0,
            }
    return result


def load_reference_partition(
    path: str | Path,
) -> tuple[dict[str, float], dict[str, int], float]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    scores = {normalize_case_id(k): float(v) for k, v in payload["scores"].items()}
    counts = {
        normalize_case_id(k): int(v)
        for k, v in payload.get("foreground_counts", {}).items()
    }
    threshold = float(payload["threshold"])
    if set(scores) != set(counts):
        raise ValueError("Reference partition needs a foreground count for every case")
    return scores, counts, threshold


def build_routing_assignments(
    case_names: list[str],
    mode: str,
    uncertainty_csv: str | Path,
    reference_partition: str | Path,
) -> tuple[list[RoutingAssignment], dict[str, float], dict[str, int | float | str]]:
    if mode not in {"original", "image", "joint", "joint_gate"}:
        raise ValueError(f"Unknown routing mode: {mode}")
    uncertainties = load_uncertainties(uncertainty_csv)
    reference_scores, foreground_counts, threshold = load_reference_partition(
        reference_partition
    )
    normalized = {normalize_case_id(name): name for name in case_names}
    expected = set(normalized)
    for label, available in (
        ("uncertainty", set(uncertainties)),
        ("reference partition", set(reference_scores)),
    ):
        if expected != available:
            missing = sorted(expected - available)
            extra = sorted(available - expected)
            raise ValueError(f"{label} case mismatch: missing={missing[:5]} extra={extra[:5]}")

    original_high = {
        case_id for case_id, score in reference_scores.items() if score > threshold
    }
    original_high_count = len(original_high)
    total = len(case_names)
    response_present = {
        case_id for case_id, count in foreground_counts.items() if count > 0
    }
    no_response = expected - response_present

    if mode == "original":
        stable = original_high
        skipped: set[str] = set()
        score_name = "cue_quality"
    else:
        skipped = no_response if mode == "joint_gate" else set()
        candidates = sorted(expected - skipped)
        # Preserve the fixed-threshold original method's observed high fraction.
        target_high = round(len(candidates) * original_high_count / total)
        score_name = "image_uncertainty" if mode == "image" else "joint_uncertainty"
        ranked = sorted(
            candidates,
            key=lambda case_id: (uncertainties[case_id][score_name], case_id),
        )
        stable = set(ranked[:target_high])

    assignments = []
    partition_scores = {}
    for case_id in sorted(expected):
        values = uncertainties[case_id]
        response_state = (
            "response-present" if case_id in response_present else "No-response"
        )
        if case_id in skipped:
            branch = "skip"
            partition_scores[normalized[case_id]] = 0.0
        elif case_id in stable:
            branch = "high"
            partition_scores[normalized[case_id]] = 1.0
        else:
            branch = "low"
            partition_scores[normalized[case_id]] = 0.0
        assignments.append(
            RoutingAssignment(
                case_id=normalized[case_id],
                response_state=response_state,
                image_uncertainty=values["image_uncertainty"],
                prompt_uncertainty=values["prompt_uncertainty"],
                joint_uncertainty=values["joint_uncertainty"],
                assigned_branch=branch,
            )
        )
    metadata: dict[str, int | float | str] = {
        "mode": mode,
        "ranking_score": score_name,
        "reference_threshold": threshold,
        "reference_high": original_high_count,
        "reference_low": total - original_high_count,
        "high": sum(row.assigned_branch == "high" for row in assignments),
        "low": sum(row.assigned_branch == "low" for row in assignments),
        "skip": sum(row.assigned_branch == "skip" for row in assignments),
        "no_response": len(no_response),
        "response_present": len(response_present),
        "no_response_all_raw_predictions_empty": all(
            foreground_counts[case_id] == 0 for case_id in no_response
        ),
        "response_present_all_raw_predictions_nonempty": all(
            foreground_counts[case_id] > 0 for case_id in response_present
        ),
        "rounding_rule": "Python round(nearest, ties-to-even)",
    }
    return assignments, partition_scores, metadata


def write_routing_csv(path: str | Path, rows: list[RoutingAssignment]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUTING_FIELDS)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
