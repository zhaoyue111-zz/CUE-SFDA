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
    candidates = []
    if best_iteration is not None:
        candidates.append(root / "validation" / f"iteration_{best_iteration:06d}" / "metrics_summary.csv")
    candidates.extend(sorted((root / "validation").glob("iteration_*/metrics_summary.csv")) if (root / "validation").is_dir() else [])
    for path in candidates:
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                return next(csv.DictReader(handle))
    evaluation = root / "evaluation" / "metrics_summary.csv"
    if evaluation.is_file():
        with evaluation.open(newline="", encoding="utf-8") as handle:
            return next(csv.DictReader(handle))
    return {}


def case_rows(root: Path, best_iteration: int | None) -> list[dict]:
    if best_iteration is None:
        return []
    path = root / "validation" / f"iteration_{best_iteration:06d}" / "metrics_per_case.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    }


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
    write_csv(args.output_dir / "ablation_results.csv", summary)
    (args.output_dir / "ablation_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

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


if __name__ == "__main__":
    main()
