#!/usr/bin/env python3
"""Build train-only routing quality and transition tables from existing CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTINGS = {
    "A_original": ROOT / "outputs/dwi_only_liver_cancer/routing.csv",
    "B_image": ROOT / "outputs/response_routing/B_image/routing.csv",
    "C_joint": ROOT / "outputs/response_routing/C_joint/routing.csv",
    "D_joint_gate": ROOT / "outputs/response_routing/D_joint_gate/routing.csv",
}
SUMMARIES = {name: path.with_name("routing_summary.json") for name, path in ROUTINGS.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(case: str) -> str:
    return case[:-7] if case.endswith(".nii.gz") else Path(case).stem


def stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {k: "" for k in ("mean", "median", "std_population", "q1", "q3")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std_population": float(np.std(array, ddof=0)),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "q3": float(np.quantile(array, 0.75, method="linear")),
    }


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uncertainty-csv",
        type=Path,
        default=Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/diagnostic/dwi_uncertainty_train/per_case_uncertainty.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis/tables")
    args = parser.parse_args()

    base_rows = read_csv(args.uncertainty_csv)
    zero_shot = {norm(row["case_id"]): float(row["baseline_dice"]) for row in base_rows}
    if len(zero_shot) != 101:
        raise ValueError(f"expected 101 zero-shot train rows; got {len(zero_shot)}")

    route_maps: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in ROUTINGS.items():
        rows = read_csv(path)
        route = {norm(row["case_id"]): row for row in rows}
        if set(route) != set(zero_shot):
            raise ValueError(f"{name} cases do not match the 101 zero-shot train cases")
        metadata = json.loads(SUMMARIES[name].read_text(encoding="utf-8"))
        if metadata.get("no_response_all_raw_predictions_empty") is not True:
            raise ValueError(f"{name}: response metadata does not support empty-mask count")
        route_maps[name] = route

    summary_rows = []
    for method, routes in route_maps.items():
        metadata = json.loads(SUMMARIES[method].read_text(encoding="utf-8"))
        group_names = ["high", "low"] + (["skip"] if method == "D_joint_gate" else [])
        grouped = {
            group: [case for case, row in routes.items() if row["assigned_branch"] == group]
            for group in group_names
        }
        high_dice = [zero_shot[c] for c in grouped["high"]]
        low_dice = [zero_shot[c] for c in grouped["low"]]
        mean_gap = float(np.mean(high_dice) - np.mean(low_dice))
        median_gap = float(np.median(high_dice) - np.median(low_dice))
        for group in group_names:
            cases = grouped[group]
            values = [zero_shot[c] for c in cases]
            empty = sum(routes[c]["response_state"] == "No-response" for c in cases)
            row = {
                "routing": method,
                "group": group,
                "n": len(cases),
                "zero_shot_dice_mean": stats(values)["mean"],
                "zero_shot_dice_median": stats(values)["median"],
                "zero_shot_dice_std_population": stats(values)["std_population"],
                "zero_shot_dice_q1": stats(values)["q1"],
                "zero_shot_dice_q3": stats(values)["q3"],
                # The frozen diagnostic CSV has no per-case TP/FP counts, so P/R are unavailable.
                "precision_mean": "",
                "recall_mean": "",
                "empty_prediction_n": empty,
                "empty_prediction_rate": empty / len(cases) if cases else "",
                "dice_lt_0_1_n": sum(v < 0.1 for v in values),
                "dice_lt_0_1_rate": sum(v < 0.1 for v in values) / len(cases) if cases else "",
                "dice_lt_0_3_n": sum(v < 0.3 for v in values),
                "dice_lt_0_3_rate": sum(v < 0.3 for v in values) / len(cases) if cases else "",
                "dice_gt_0_5_n": sum(v > 0.5 for v in values),
                "dice_gt_0_5_rate": sum(v > 0.5 for v in values) / len(cases) if cases else "",
                "dice_gt_0_7_n": sum(v > 0.7 for v in values),
                "dice_gt_0_7_rate": sum(v > 0.7 for v in values) / len(cases) if cases else "",
                "high_minus_low_mean_dice_gap": mean_gap,
                "high_minus_low_median_dice_gap": median_gap,
                "high_poor_dice_lt_0_1_n": sum(v < 0.1 for v in high_dice),
                "high_poor_dice_lt_0_1_rate": sum(v < 0.1 for v in high_dice) / len(high_dice) if high_dice else "",
                "low_good_dice_gt_0_5_n": sum(v > 0.5 for v in low_dice),
                "low_good_dice_gt_0_5_rate": sum(v > 0.5 for v in low_dice) / len(low_dice) if low_dice else "",
                "low_good_dice_gt_0_7_n": sum(v > 0.7 for v in low_dice),
                "low_good_dice_gt_0_7_rate": sum(v > 0.7 for v in low_dice) / len(low_dice) if low_dice else "",
                "empty_count_basis": "response_state=No-response; routing_summary confirms all such raw predictions empty",
                "routing_high": metadata.get("high", ""),
                "routing_low": metadata.get("low", ""),
                "routing_skip": metadata.get("skip", ""),
            }
            summary_rows.append(row)
    write(args.out_dir / "step2_routing_quality_summary.csv", summary_rows)

    transition_rows = []
    for first, second, label in [
        ("A_original", "B_image", "Original_to_Image"),
        ("A_original", "C_joint", "Original_to_Joint"),
        ("B_image", "C_joint", "Image_to_Joint"),
    ]:
        for source in ("high", "low"):
            for target in ("high", "low"):
                cases = sorted(
                    case for case in zero_shot
                    if route_maps[first][case]["assigned_branch"] == source
                    and route_maps[second][case]["assigned_branch"] == target
                )
                values = [zero_shot[c] for c in cases]
                s = stats(values)
                transition_rows.append({
                    "transition": label,
                    "source_routing": first,
                    "target_routing": second,
                    "source_group": source,
                    "target_group": target,
                    "identity_changed": source != target,
                    "n": len(cases),
                    "fraction_of_101": len(cases) / 101,
                    "zero_shot_dice_mean": s["mean"],
                    "zero_shot_dice_median": s["median"],
                    "zero_shot_dice_std_population": s["std_population"],
                    "zero_shot_dice_q1": s["q1"],
                    "zero_shot_dice_q3": s["q3"],
                    "case_ids": ";".join(cases),
                })
    write(args.out_dir / "step2_routing_transition.csv", transition_rows)

    case_rows = []
    for case, dice in sorted(zero_shot.items()):
        case_rows.append({
            "case_id": case + ".nii.gz",
            "zero_shot_dice": dice,
            **{f"{method}_group": route_maps[method][case]["assigned_branch"] for method in route_maps},
            **{f"{method}_response_state": route_maps[method][case]["response_state"] for method in route_maps},
        })
    write(args.out_dir / "step2_routing_case_join.csv", case_rows)

    print(f"wrote {args.out_dir / 'step2_routing_quality_summary.csv'} ({len(summary_rows)} group rows)")
    print(f"wrote {args.out_dir / 'step2_routing_transition.csv'} ({len(transition_rows)} transition cells)")
    print(f"wrote {args.out_dir / 'step2_routing_case_join.csv'} ({len(case_rows)} train cases)")


if __name__ == "__main__":
    main()
