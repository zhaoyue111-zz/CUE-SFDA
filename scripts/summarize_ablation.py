#!/usr/bin/env python3
"""Summarize the CUE-SFDA A--H ablation lineage without using GT for training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def best_validation(root: Path, best_iteration: int | None) -> dict:
    if best_iteration is not None:
        best_path = root / "validation" / f"iteration_{best_iteration:06d}" / "metrics_summary.csv"
        if best_path.is_file():
            with best_path.open(newline="", encoding="utf-8") as handle:
                return next(csv.DictReader(handle), {})
    validation_paths = sorted((root / "validation").glob("iteration_*/metrics_summary.csv")) if (root / "validation").is_dir() else []
    rows = []
    for path in validation_paths:
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                item = next(csv.DictReader(handle), {})
            if item:
                rows.append(item)
    if rows:
        return max(rows, key=lambda item: float(item.get("mean_dice", "-inf")))
    for evaluation in (
        root / "evaluation" / "metrics_summary.csv",
        root / "best_evaluation" / "evaluation" / "metrics_summary.csv",
    ):
        if evaluation.is_file():
            with evaluation.open(newline="", encoding="utf-8") as handle:
                return next(csv.DictReader(handle))
    return {}


def case_rows(root: Path, best_iteration: int | None) -> list[dict]:
    if best_iteration is not None:
        best_path = root / "validation" / f"iteration_{best_iteration:06d}" / "metrics_per_case.csv"
        if best_path.is_file():
            with best_path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
    paths = sorted((root / "validation").glob("iteration_*/metrics_per_case.csv")) if (root / "validation").is_dir() else []
    best_rows = []
    best_dice = float("-inf")
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        mean_dice = sum(float(item["dice"]) for item in rows) / max(len(rows), 1)
        if mean_dice > best_dice:
            best_dice, best_rows = mean_dice, rows
    return best_rows


def row(name: str, root: Path) -> dict:
    early = read_json(root / "early_stopping.json")
    refresh = read_json(root / "refresh_history.json")
    best_iteration = early.get("best_iteration")
    metrics = best_validation(root, best_iteration)
    return {
        "Method": name,
        "Best Iter": best_iteration if best_iteration is not None else "",
        "Stop Iter": early.get("stop_iteration", ""),
        "Dice": metrics.get("mean_dice", ""),
        "mIoU": metrics.get("mean_iou", ""),
        "Precision": metrics.get("mean_precision", ""),
        "Recall": metrics.get("mean_recall", ""),
        "Refresh Cycles": len(refresh.get("refresh_history", [])),
        "Foreground Volume Change": refresh.get("refresh_history", [{}])[-1].get("mean_foreground_volume_change", "") if refresh.get("refresh_history") else "",
    }


def validation_at(root: Path, iteration: int) -> dict:
    path = root / "validation" / f"iteration_{iteration:06d}" / "metrics_summary.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), {})


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()

    experiments = [("C baseline", args.baseline)]
    for item in args.experiment:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"Invalid --experiment {item!r}; expected NAME=PATH")
        experiments.append((name, Path(path)))

    summary = [row(name, root) for name, root in experiments]
    baseline_metrics = best_validation(args.baseline, read_json(args.baseline / "early_stopping.json").get("best_iteration"))
    for item in summary[1:]:
        precision_delta = float(item.get("Precision") or 0.0) - float(baseline_metrics.get("mean_precision", 0.0))
        recall_delta = float(item.get("Recall") or 0.0) - float(baseline_metrics.get("mean_recall", 0.0))
        item["Diagnosis"] = "foreground expansion dominant" if recall_delta > 0 and precision_delta < 0 else ""
    summary[0]["Diagnosis"] = ""
    write_csv(args.output_dir / "ablation_results.csv", summary)
    (args.output_dir / "ablation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    baseline_rows = case_rows(args.baseline, None)
    if not baseline_rows:
        evaluation = args.baseline / "evaluation" / "metrics_per_case.csv"
        if evaluation.is_file():
            with evaluation.open(newline="", encoding="utf-8") as handle:
                baseline_rows = list(csv.DictReader(handle))
    bins = {"<0.1": [], "0.1-0.5": [], ">0.5": []}
    for item in baseline_rows:
        dice = float(item["dice"])
        key = "<0.1" if dice < 0.1 else "0.1-0.5" if dice <= 0.5 else ">0.5"
        bins[key].append(item)
    subgroup = []
    for key, items in bins.items():
        subgroup.append({
            "subgroup": key,
            "n": len(items),
            "mean_dice": sum(float(x["dice"]) for x in items) / len(items) if items else "",
            "mean_precision": sum(float(x["precision"]) for x in items) / len(items) if items else "",
            "mean_recall": sum(float(x["recall"]) for x in items) / len(items) if items else "",
        })
    write_csv(args.output_dir / "baseline_dice_subgroups.csv", subgroup)

    baseline_by_case = {item.get("case"): item for item in baseline_rows}
    comparison_rows = []
    refresh_analysis = []
    for name, root in experiments:
        if name == "C baseline":
            continue
        best = read_json(root / "early_stopping.json").get("best_iteration")
        method_rows = case_rows(root, best)
        method_by_case = {item.get("case"): item for item in method_rows}
        improved = degraded = 0
        for case, baseline in baseline_by_case.items():
            if case not in method_by_case:
                continue
            delta = float(method_by_case[case]["dice"]) - float(baseline["dice"])
            improved += int(delta > 0)
            degraded += int(delta < 0)
        comparison_rows.append({"method": name, "improved_cases": improved, "degraded_cases": degraded})
        method_metrics = best_validation(root, best)
        precision_delta = float(method_metrics.get("mean_precision", 0.0)) - float(baseline_metrics.get("mean_precision", 0.0))
        recall_delta = float(method_metrics.get("mean_recall", 0.0)) - float(baseline_metrics.get("mean_recall", 0.0))
        diagnosis = "foreground expansion dominant" if recall_delta > 0 and precision_delta < 0 else ""
        history = read_json(root / "refresh_history.json").get("refresh_history", [])
        for event in history:
            refresh_analysis.append({
                "method": name,
                "refresh_cycle": event.get("refresh_cycle", ""),
                "trigger_iteration": event.get("trigger_iteration", ""),
                "covered_cases": event.get("covered_cases", ""),
                "dynamic_added_foreground_voxels": event.get("dynamic_added_foreground_voxels", ""),
                "dynamic_removed_foreground_voxels": event.get("dynamic_removed_foreground_voxels", ""),
                "mean_foreground_volume_change": event.get("mean_foreground_volume_change", ""),
                "mean_confidence_change": event.get("mean_confidence_change", ""),
                "accepted_voxel_ratio": event.get("accepted_voxel_ratio", ""),
                "rejected_voxel_ratio": event.get("rejected_voxel_ratio", ""),
                "precision_delta_vs_baseline": precision_delta,
                "recall_delta_vs_baseline": recall_delta,
                "diagnosis": diagnosis,
            })
    write_csv(args.output_dir / "case_comparisons.csv", comparison_rows)
    write_csv(args.output_dir / "refresh_analysis.csv", refresh_analysis)


if __name__ == "__main__":
    main()
