#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def final_metrics_path(directory: Path) -> Path:
    for subdir in ("best_evaluation", "evaluation"):
        # The CLI writes <output-dir>/evaluation/metrics_per_case.csv.
        for candidate in (
            directory / subdir / "metrics_per_case.csv",
            directory / subdir / "evaluation" / "metrics_per_case.csv",
        ):
            if candidate.is_file():
                return candidate
    candidates = sorted((directory / "validation").glob("iteration_*/metrics_per_case.csv"))
    if not candidates:
        raise FileNotFoundError(f"No final metrics found under {directory}")
    return candidates[-1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_name(dice: float) -> str:
    if dice < 0.1:
        return "<0.1"
    if dice <= 0.5:
        return "0.1-0.5"
    return ">0.5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--experiment", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-baseline-dice", type=float, default=0.283336)
    args = parser.parse_args()

    baseline_rows = read_rows(final_metrics_path(args.baseline_dir))
    baseline = {row["case"]: row for row in baseline_rows}
    baseline_dice = float(np.mean([float(row["dice"]) for row in baseline_rows]))
    if abs(baseline_dice - args.expected_baseline_dice) > 5e-6:
        raise RuntimeError(
            f"Baseline mismatch: {baseline_dice:.9f} vs {args.expected_baseline_dice:.9f}"
        )
    if any(int(row["ground_truth_voxels"]) == 0 for row in baseline_rows):
        raise RuntimeError("Empty test ground truth detected")

    overall, subgroup_rows, sanity = [], [], {
        "baseline_mean_dice": baseline_dice,
        "baseline_matches_existing": True,
        "empty_test_ground_truth": 0,
        "experiments": {},
    }
    labels, dices = [], []
    reference_config = None
    for name, directory_text in args.experiment:
        directory = Path(directory_text)
        adapted_rows = read_rows(final_metrics_path(directory))
        adapted = {row["case"]: row for row in adapted_rows}
        if set(adapted) != set(baseline):
            raise RuntimeError(f"{name}: test cases differ from baseline")
        per_case = []
        for case_id, base in baseline.items():
            row = adapted[case_id]
            base_dice, adapted_dice = float(base["dice"]), float(row["dice"])
            per_case.append({
                "case_id": case_id,
                "baseline_dice": base_dice,
                "adapted_dice": adapted_dice,
                "delta_dice": adapted_dice - base_dice,
                "baseline_group": group_name(base_dice),
            })
        write_csv(directory / "test_case_deltas.csv", per_case)
        summary = {
            "experiment": name,
            "dice": float(np.mean([float(row["dice"]) for row in adapted_rows])),
            "mIoU": float(np.mean([float(row["iou"]) for row in adapted_rows])),
            "precision": float(np.mean([float(row["precision"]) for row in adapted_rows])),
            "recall": float(np.mean([float(row["recall"]) for row in adapted_rows])),
        }
        summary["dice_vs_original"] = 0.0
        overall.append(summary)
        for group in ("<0.1", "0.1-0.5", ">0.5"):
            selected = [row for row in per_case if row["baseline_group"] == group]
            subgroup_rows.append({
                "experiment": name,
                "baseline_group": group,
                "count": len(selected),
                "baseline_mean_dice": float(np.mean([row["baseline_dice"] for row in selected])),
                "adapted_mean_dice": float(np.mean([row["adapted_dice"] for row in selected])),
                "mean_delta_dice": float(np.mean([row["delta_dice"] for row in selected])),
            })
        routing = json.load((directory / "routing_summary.json").open())
        resolved = json.load((directory / "resolved_config.json").open())
        comparable = {
            "data": resolved["data"],
            "model": resolved["model"],
            "train": resolved["train"],
            "evaluation": resolved["evaluation"],
            "method": {
                key: value for key, value in resolved["method"].items()
                if not key.startswith("routing_")
            },
        }
        if reference_config is None:
            reference_config = comparable
        elif comparable != reference_config:
            raise RuntimeError(f"{name}: non-routing configuration differs from A")
        route_rows = read_rows(directory / "routing.csv")
        no_response = [row for row in route_rows if row["response_state"] == "No-response"]
        if len(no_response) != int(routing["no_response"]):
            raise RuntimeError(f"{name}: No-response count mismatch")
        if name == "D" and any(row["assigned_branch"] != "skip" for row in no_response):
            raise RuntimeError("D: a No-response case was not skipped")
        log_text = (directory / "run.log").read_text(encoding="utf-8")
        losses = [float(value) for value in re.findall(r"loss=([0-9.eE+-]+)", log_text)]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise RuntimeError(f"{name}: missing/non-finite logged losses")
        if not any(value > 0 for value in losses):
            raise RuntimeError(f"{name}: training loss is always zero")
        nonempty_predictions = sum(int(row["predicted_voxels"]) > 0 for row in adapted_rows)
        if nonempty_predictions == 0:
            raise RuntimeError(f"{name}: model collapsed to empty predictions")
        sanity["experiments"][name] = {
            "routing": routing,
            "non_routing_config_matches_original": True,
            "finite_losses": True,
            "loss_not_always_zero": True,
            "logged_loss_min": min(losses),
            "logged_loss_max": max(losses),
            "nonempty_test_predictions": nonempty_predictions,
            "empty_test_predictions": len(adapted_rows) - nonempty_predictions,
        }
        labels.append(name)
        dices.append(summary["dice"])

    original = next(row["dice"] for row in overall if row["experiment"] == "A")
    for row in overall:
        row["dice_vs_original"] = row["dice"] - original
    write_csv(args.output_dir / "overall_metrics.csv", overall)
    write_csv(args.output_dir / "baseline_dice_subgroups.csv", subgroup_rows)
    with (args.output_dir / "sanity_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(sanity, handle, indent=2, ensure_ascii=False)
    best = max(overall, key=lambda row: row["dice"])
    with (args.output_dir / "result_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_experiment": best["experiment"], "overall": overall}, handle, indent=2)

    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    bars = axis.bar(labels, dices, color=["#777777", "#4C78A8", "#F58518", "#54A24B"])
    axis.axhline(baseline_dice, color="black", linestyle="--", linewidth=1, label="VoxTell baseline")
    axis.set_ylabel("Mean Dice")
    axis.set_title("Response-aware uncertainty routing")
    axis.legend()
    for bar, value in zip(bars, dices):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    figure.savefig(args.output_dir / "overall_dice.png", dpi=180)


if __name__ == "__main__":
    main()
