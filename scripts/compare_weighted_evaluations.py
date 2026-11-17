from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["case"]: row for row in rows}


def read_summary(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {key: float(value) for key, value in row.items() if key != "num_cases"} | {
        "num_cases": float(row["num_cases"])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="Evaluation directory")
    parser.add_argument("--weighted", type=Path, required=True, help="Evaluation directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_rows(args.baseline / "metrics_per_case.csv")
    weighted = read_rows(args.weighted / "metrics_per_case.csv")
    if set(baseline) != set(weighted):
        raise ValueError("Baseline and weighted evaluations do not contain the same cases")
    rows = []
    deltas = []
    for case in sorted(baseline):
        old, new = float(baseline[case]["dice"]), float(weighted[case]["dice"])
        delta = new - old
        deltas.append(delta)
        rows.append({"case": case, "baseline_dice": old, "weighted_dice": new, "delta_dice": delta})
    values = np.asarray(deltas, dtype=np.float64)
    old_summary = read_summary(args.baseline / "metrics_summary.csv")
    new_summary = read_summary(args.weighted / "metrics_summary.csv")
    metrics = {}
    for key in ("mean_dice", "mean_iou", "mean_precision", "mean_recall"):
        metrics[key] = {
            "baseline": old_summary[key],
            "weighted": new_summary[key],
            "delta": new_summary[key] - old_summary[key],
        }
    report = {
        "num_cases": len(rows),
        "metrics": metrics,
        "case_dice_delta": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "improved_cases": int((values > 0).sum()),
            "degraded_cases": int((values < 0).sum()),
            "unchanged_cases": int((values == 0).sum()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case_dice_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
