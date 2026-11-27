from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("dice", "iou", "precision", "recall", "hd95", "asd", "predicted_voxels")
STRATA = ("Dice<0.1", "0.1<=Dice<=0.5", "Dice>0.5")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_summary(rows: list[dict]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for metric in METRICS:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric, "") not in ("", "nan", "NaN")
        ]
        if values:
            output[f"mean_{metric}"] = float(np.mean(values))
    if rows:
        output["num_cases"] = len(rows)
        output["empty_predictions"] = sum(
            int(float(row.get("predicted_voxels", 0)) == 0) for row in rows
        )
    return output


def latest_validation(root: Path) -> tuple[int, Path] | None:
    candidates = []
    for directory in (root / "validation").glob("iteration_*"):
        per_case = directory / "metrics_per_case.csv"
        if per_case.is_file():
            try:
                iteration = int(directory.name.rsplit("_", 1)[1])
            except ValueError:
                continue
            candidates.append((iteration, per_case))
    return max(candidates, default=None, key=lambda item: item[0])


def best_checkpoint(root: Path) -> tuple[int, Path] | None:
    candidates = []
    for path in (root / "checkpoints").glob("best_iteration_*.pt"):
        match = re.search(r"best_iteration_(\d+)_dice_", path.name)
        if match:
            iteration = int(match.group(1))
            metrics = root / "validation" / f"iteration_{iteration:06d}" / "metrics_per_case.csv"
            if metrics.is_file():
                candidates.append((iteration, metrics))
    return max(candidates, default=None, key=lambda item: item[0])


def optimizer_step_count(root: Path, iteration: int) -> int | None:
    candidates = [root / "checkpoints" / "latest.pt"] + list(
        (root / "checkpoints").glob(f"best_iteration_{iteration:06d}_dice_*.pt")
    )
    path = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_file()
            and int(torch.load(candidate, map_location="cpu", weights_only=False).get("iteration", -1))
            == iteration
        ),
        None,
    )
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("trainer_state", {})
    count = state.get("optimizer_step_count")
    if count is not None and int(checkpoint.get("iteration", -1)) == iteration:
        return int(count)
    optimizer_steps = {
        int(value["step"].item() if hasattr(value["step"], "item") else value["step"])
        for value in checkpoint.get("optimizer_state", {}).get("state", {}).values()
        if "step" in value
    }
    if len(optimizer_steps) == 1 and int(checkpoint.get("iteration", -1)) == iteration:
        return next(iter(optimizer_steps))
    return None


def with_diagnostics(
    rows: list[dict],
    initial_dice: dict[str, float],
    initial_volume: dict[str, float],
) -> tuple[dict, list[dict]]:
    enriched = []
    for row in rows:
        case = row["case"]
        base_dice = initial_dice.get(case)
        base_volume = initial_volume.get(case)
        current_volume = float(row.get("predicted_voxels", 0))
        delta = float(row["dice"]) - base_dice if base_dice is not None else np.nan
        enriched.append(
            {
                **row,
                "initial_zero_shot_dice": base_dice,
                "initial_dice_stratum": (
                    "unknown" if base_dice is None else
                    STRATA[0] if base_dice < 0.1 else
                    STRATA[1] if base_dice <= 0.5 else STRATA[2]
                ),
                "dice_change_vs_zero_shot": delta,
                "initial_predicted_voxels": base_volume,
                "predicted_voxels_change": (
                    current_volume - base_volume if base_volume is not None else np.nan
                ),
                "new_empty_prediction": int(
                    base_volume is not None and base_volume > 0 and current_volume == 0
                ),
            }
        )
    deltas = [float(row["dice_change_vs_zero_shot"]) for row in enriched]
    summary = {
        **metric_summary(enriched),
        "improved_over_zero_shot_gt_0_02": sum(value > 0.02 for value in deltas),
        "decreased_over_zero_shot_gt_0_02": sum(value < -0.02 for value in deltas),
        "new_empty_predictions_vs_zero_shot": sum(
            int(row["new_empty_prediction"]) for row in enriched
        ),
    }
    return summary, enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize hierarchical SFDA validation, best/final checkpoints, and initial-Dice strata"
    )
    parser.add_argument("--output", type=Path, default=Path("analysis/hierarchical_dynamic"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    zero_shot_path = ROOT / "outputs/hierarchical_dynamic/M1/validation/iteration_000000/metrics_per_case.csv"
    if not zero_shot_path.is_file():
        raise FileNotFoundError(f"Initial source-model validation missing: {zero_shot_path}")
    baseline = read_rows(zero_shot_path)
    initial_dice = {row["case"]: float(row["dice"]) for row in baseline}
    initial_volume = {row["case"]: float(row["predicted_voxels"]) for row in baseline}
    strata = {
        case: STRATA[0] if value < 0.1 else STRATA[1] if value <= 0.5 else STRATA[2]
        for case, value in initial_dice.items()
    }
    summary_rows = []
    per_case_rows = []
    baseline_summary, baseline_enriched = with_diagnostics(
        baseline, initial_dice, initial_volume
    )
    summary_rows.append(
        {"experiment": "Zero-shot", "checkpoint_selection": "initial", "checkpoint": "iteration_000000", **baseline_summary}
    )
    for stratum in STRATA:
        group = [row for row in baseline_enriched if row["initial_dice_stratum"] == stratum]
        if group:
            summary_rows.append(
                {"experiment": "Zero-shot", "checkpoint_selection": "initial", "checkpoint": "iteration_000000", "initial_dice_stratum": stratum, **metric_summary(group)}
            )
    for row in baseline_enriched:
        per_case_rows.append(
            {"experiment": "Zero-shot", "checkpoint_selection": "initial", "checkpoint_iteration": 0, **row}
        )

    experiments = {
        "M0_aligned_dynamic_hard": ROOT / "outputs/hierarchical_dynamic/M0_aligned_dynamic_hard",
        "M1": ROOT / "outputs/hierarchical_dynamic/M1",
        "M2": ROOT / "outputs/hierarchical_dynamic/M2",
        "M3-control": ROOT / "outputs/hierarchical_dynamic/M3-control",
        "M3-discount": ROOT / "outputs/hierarchical_dynamic/M3-discount",
        "M4": ROOT / "outputs/hierarchical_dynamic/M4",
        "M5": ROOT / "outputs/hierarchical_dynamic/M5",
        "M6": ROOT / "outputs/hierarchical_dynamic/M6",
    }
    for name, root in experiments.items():
        selections = []
        latest = latest_validation(root)
        best = best_checkpoint(root)
        if latest is not None:
            selections.append(("final_validated", latest))
        if best is not None and all(best[0] != item[1][0] for item in selections):
            selections.append(("best", best))
        training_summary_path = root / "training_summary.json"
        training_summary = {}
        if training_summary_path.is_file():
            import json

            training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        for selection_name, (iteration, path) in selections:
            rows = read_rows(path)
            summary, enriched = with_diagnostics(rows, initial_dice, initial_volume)
            fixed_budget = iteration == 2000
            summary_rows.append(
                {
                    "experiment": name,
                    "checkpoint_selection": selection_name,
                    "checkpoint": f"iteration_{iteration:06d}",
                    "fixed_budget_2000": int(fixed_budget),
                    "optimizer_steps": optimizer_step_count(root, iteration),
                    "training_iterations_completed": training_summary.get("iterations_completed"),
                    **summary,
                }
            )
            for stratum in STRATA:
                group = [row for row in enriched if row["initial_dice_stratum"] == stratum]
                if group:
                    group_summary = metric_summary(group)
                    group_deltas = [float(row["dice_change_vs_zero_shot"]) for row in group]
                    summary_rows.append(
                        {
                            "experiment": name,
                            "checkpoint_selection": selection_name,
                            "checkpoint": f"iteration_{iteration:06d}",
                            "fixed_budget_2000": int(fixed_budget),
                            "initial_dice_stratum": stratum,
                            **group_summary,
                            "improved_over_zero_shot_gt_0_02": sum(value > 0.02 for value in group_deltas),
                            "decreased_over_zero_shot_gt_0_02": sum(value < -0.02 for value in group_deltas),
                            "new_empty_predictions_vs_zero_shot": sum(int(row["new_empty_prediction"]) for row in group),
                        }
                    )
            for row in enriched:
                per_case_rows.append(
                    {
                        "experiment": name,
                        "checkpoint_selection": selection_name,
                        "checkpoint_iteration": iteration,
                        **row,
                    }
                )

    controls = {
        "M0_historical_E_joint": ROOT / "outputs/response_routing/E_joint_empty_to_low/validation/iteration_001800/metrics_per_case.csv",
        "M0_historical_CSS": ROOT / "outputs/high_selection/CSS/validation/iteration_002000/metrics_per_case.csv",
    }
    for name, path in controls.items():
        if not path.is_file():
            continue
        rows = read_rows(path)
        summary, enriched = with_diagnostics(rows, initial_dice, initial_volume)
        summary_rows.append(
            {"experiment": name, "checkpoint_selection": "historical_protocol", "checkpoint": path.parent.name, **summary}
        )
        for row in enriched:
            per_case_rows.append(
                {"experiment": name, "checkpoint_selection": "historical_protocol", "checkpoint_iteration": int(path.parent.name.rsplit("_", 1)[1]), **row}
            )

    for filename, rows in (
        ("validation_summary.csv", summary_rows),
        ("validation_per_case.csv", per_case_rows),
    ):
        if not rows:
            continue
        fields = sorted({key for row in rows for key in row})
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(summary_rows)} validation summary rows under {output}")


if __name__ == "__main__":
    main()
