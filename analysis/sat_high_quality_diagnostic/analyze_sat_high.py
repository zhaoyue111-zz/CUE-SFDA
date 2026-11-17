#!/usr/bin/env python3
"""Post-hoc SAT vs frozen VoxTell analysis for the fixed C_joint High set."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).resolve().parent
IMAGE_DIR = Path("/data/hilab/datasets/WCH_MRI145/DWI/images")
GT_DIR = Path("/data/hilab/datasets/WCH_MRI145/DWI/labels")
SPLIT = Path("/data/hilab/datasets/WCH_MRI145/DWI/split.json")
ROUTING = ROOT / "outputs/response_routing/C_joint/routing.csv"
FROZEN_DICE = Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/diagnostic/dwi_uncertainty_train/per_case_uncertainty.csv")
OLD_SAT = ROOT / "outputs/response_routing/sat_no_response/results/D_no_response"
OLD_SAT_METRICS = ROOT / "outputs/response_routing/sat_no_response/diagnostic/per_case.csv"
NEW_SAT = ANALYSIS / "sat_predictions/C_joint_high"
VOXTELL_PROB = ROOT / "outputs/response_routing/sat_no_response/voxtell_response/voxtell_probability"

spec = importlib.util.spec_from_file_location(
    "sat_no_response_diagnostic", ROOT / "scripts/sat_no_response_diagnostic.py"
)
sat_diag = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(sat_diag)
transform_gt = sat_diag.transform_gt
metric = sat_diag.metric


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def key(case_id: str) -> str:
    return case_id[:-7] if case_id.endswith(".nii.gz") else Path(case_id).stem


def nii_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj) > 0.5


def sat_path(case_id: str) -> tuple[Path, str]:
    stem = key(case_id)
    old = OLD_SAT / f"seg_{stem}" / "liver cancer.nii.gz"
    if old.is_file():
        return old, "reused_sat_no_response"
    new = NEW_SAT / f"seg_{stem}" / "liver cancer.nii.gz"
    if new.is_file():
        return new, "new_sat_nano_inference"
    raise FileNotFoundError(f"SAT prediction missing for {case_id}: expected {old} or {new}")


def divide(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def metrics(mask: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    return metric(mask, gt)


def mean(rows: list[dict], field: str) -> float | None:
    values = [float(r[field]) for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))]
    return float(np.mean(values)) if values else None


def median(rows: list[dict], field: str) -> float | None:
    values = [float(r[field]) for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))]
    return float(np.median(values)) if values else None


def count(rows: list[dict], predicate) -> int:
    return sum(bool(predicate(r)) for r in rows)


def summarize(group: str, rows: list[dict]) -> dict:
    complete = [r for r in rows if r["overlap_metrics_available"]]
    deltas = [float(r["sat_minus_voxtell_dice"]) for r in rows]
    return {
        "subgroup": group,
        "n": len(rows),
        "voxtell_mask_n": count(rows, lambda r: r["voxtell_prediction_available"]),
        "sat_mask_n": count(rows, lambda r: r["sat_prediction_available"]),
        "overlap_mask_n": len(complete),
        "voxtell_empty_n": count(rows, lambda r: r["voxtell_empty"]),
        "sat_nonempty_n": count(rows, lambda r: r["sat_foreground_voxels"] > 0),
        "sat_nonempty_rate": divide(count(rows, lambda r: r["sat_foreground_voxels"] > 0), len(rows)),
        "voxtell_dice_mean": mean(rows, "voxtell_dice"),
        "voxtell_dice_median": median(rows, "voxtell_dice"),
        "voxtell_precision_mean_available": mean(rows, "voxtell_precision"),
        "voxtell_recall_mean_available": mean(rows, "voxtell_recall"),
        "voxtell_foreground_voxels_mean_available": mean(rows, "voxtell_foreground_voxels"),
        "sat_dice_mean": mean(rows, "sat_dice"),
        "sat_dice_median": median(rows, "sat_dice"),
        "sat_precision_mean": mean(rows, "sat_precision"),
        "sat_recall_mean": mean(rows, "sat_recall"),
        "sat_foreground_voxels_mean": mean(rows, "sat_foreground_voxels"),
        "gt_foreground_voxels_mean": mean(rows, "gt_foreground_voxels"),
        "sat_minus_voxtell_dice_mean": float(np.mean(deltas)) if deltas else None,
        "sat_minus_voxtell_dice_median": float(np.median(deltas)) if deltas else None,
        "sat_dice_improved_n": sum(x > 0 for x in deltas),
        "sat_dice_degraded_n": sum(x < 0 for x in deltas),
        "delta_dice_gt_0_02_n": sum(x > 0.02 for x in deltas),
        "delta_dice_gt_0_05_n": sum(x > 0.05 for x in deltas),
        "delta_dice_lt_minus_0_02_n": sum(x < -0.02 for x in deltas),
        "delta_dice_lt_minus_0_05_n": sum(x < -0.05 for x in deltas),
        "best_of_two_dice_mean": mean(rows, "best_of_two_dice"),
        "best_of_two_mean_uplift": mean(rows, "best_of_two_uplift"),
        "best_of_two_median_uplift": median(rows, "best_of_two_uplift"),
        "best_of_two_improved_n": count(rows, lambda r: r["best_of_two_uplift"] > 0),
        "sat_dice_gt_0_1_n": count(rows, lambda r: r["sat_dice"] > 0.1),
        "sat_dice_gt_0_3_n": count(rows, lambda r: r["sat_dice"] > 0.3),
        "sat_dice_gt_0_5_n": count(rows, lambda r: r["sat_dice"] > 0.5),
        "sat_dice_lt_0_1_n": count(rows, lambda r: r["sat_dice"] < 0.1),
        "sat_only_precision_mean_available": mean(complete, "sat_only_precision"),
        "voxtell_only_precision_mean_available": mean(complete, "voxtell_only_precision"),
        "intersection_dice_mean_available": mean(complete, "intersection_dice"),
        "intersection_precision_mean_available": mean(complete, "intersection_precision"),
        "intersection_recall_mean_available": mean(complete, "intersection_recall"),
        "union_dice_mean_available": mean(complete, "union_dice"),
        "union_precision_mean_available": mean(complete, "union_precision"),
        "union_recall_mean_available": mean(complete, "union_recall"),
    }


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train = {Path(x).name for x in split.get("train_cases", split.get("train"))}
    routes = csv_rows(ROUTING)
    high = [r for r in routes if r["assigned_branch"].lower() == "high"]
    if len(train) != 101 or len(high) != 59 or any(r["case_id"] not in train for r in high):
        raise ValueError("C_joint High must be exactly 59 cases from the fixed 101-case train split")
    zero_shot = {key(r["case_id"]): float(r["baseline_dice"]) for r in csv_rows(FROZEN_DICE)}
    prob_cases = {p.name for p in VOXTELL_PROB.glob("*.nii.gz")}
    old_sat_metrics = {r["case_id"]: r for r in csv_rows(OLD_SAT_METRICS)}

    rows: list[dict] = []
    for route in sorted(high, key=lambda r: r["case_id"]):
        case_id = route["case_id"]
        case_key = key(case_id)
        if case_key not in zero_shot:
            raise KeyError(f"No frozen zero-shot Dice for {case_id}")
        v_dice = zero_shot[case_key]
        empty = route["response_state"] == "No-response"
        subgroup = "Empty High" if empty else ("Non-empty Hard High" if v_dice < 0.5 else "Good High")
        if empty and case_id not in prob_cases:
            raise FileNotFoundError(f"Expected saved frozen probability for Empty High {case_id}")
        if not empty and v_dice == 0:
            # The fixed routing's response-present status establishes a non-empty raw mask;
            # direct mask files are still tracked separately below.
            pass

        image_path = IMAGE_DIR / case_id
        sat_file, sat_source = sat_path(case_id)
        sat_mask = nii_mask(sat_file)
        if sat_source == "reused_sat_no_response":
            # Reuse the metrics computed with the original SAT preprocessing
            # environment. Its saved mask grid differs by one voxel from the
            # current MONAI transform for some cases; no mask alignment is
            # needed because the paired frozen VoxTell mask is empty.
            old = old_sat_metrics[case_id]
            sat_dice, sat_p, sat_r = (float(old[k]) for k in ("dice", "precision", "recall"))
            sat_volume = int(old["sat_prediction_voxels"])
            gt_volume = int(old["gt_voxels"])
            gt = None
            sat_mask = None
        else:
            gt = transform_gt(image_path, GT_DIR / case_id)
            if sat_mask.shape != gt.shape:
                raise ValueError(f"SAT/GT grid mismatch for {case_id}: {sat_mask.shape} vs {gt.shape}")
            sat_dice, sat_p, sat_r = metrics(sat_mask, gt)
            sat_volume = int(sat_mask.sum())
            gt_volume = int(gt.sum())

        # Only the 10 already-saved Empty High VoxTell masks are available.
        # Routing response_state agrees with the saved zero-shot empty masks.
        v_available = empty
        if v_available:
            if np.any(nii_mask(VOXTELL_PROB / case_id)):
                raise ValueError(f"Saved frozen mask contradicts No-response state for {case_id}")

        rec = {
            "case_id": case_id,
            "response_state": route["response_state"],
            "subgroup": subgroup,
            "voxtell_empty": empty,
            "voxtell_prediction_available": v_available,
            "sat_prediction_available": True,
            "sat_prediction_source": sat_source,
            "voxtell_dice": v_dice,
            "voxtell_precision": None,
            "voxtell_recall": None,
            "voxtell_foreground_voxels": None,
            "sat_dice": sat_dice,
            "sat_precision": sat_p,
            "sat_recall": sat_r,
            "sat_foreground_voxels": sat_volume,
            "gt_foreground_voxels": gt_volume,
            "intersection_dice": None,
            "intersection_precision": None,
            "intersection_recall": None,
            "union_dice": None,
            "union_precision": None,
            "union_recall": None,
            "sat_only_precision": None,
            "voxtell_only_precision": None,
            "overlap_metrics_available": v_available,
            "sat_minus_voxtell_dice": sat_dice - v_dice,
            "best_of_two_dice": max(v_dice, sat_dice),
            "best_of_two_uplift": max(v_dice, sat_dice) - v_dice,
        }
        if v_available:
            rec["voxtell_precision"] = 0.0
            rec["voxtell_recall"] = 0.0
            rec["voxtell_foreground_voxels"] = 0
            # M_V is empty: intersection and VoxTell-only are empty, while
            # union and SAT-only are exactly the saved SAT mask.
            rec["intersection_dice"] = 0.0 if gt_volume else 1.0
            rec["intersection_precision"] = 0.0
            rec["intersection_recall"] = 0.0
            rec["union_dice"], rec["union_precision"], rec["union_recall"] = sat_dice, sat_p, sat_r
            rec["sat_only_precision"] = sat_p
            rec["voxtell_only_precision"] = 0.0
        rows.append(rec)

    subgroup_rows = [summarize(name, [r for r in rows if r["subgroup"] == name]) for name in
                     ("Empty High", "Non-empty Hard High", "Good High")]
    subgroup_rows.append(summarize("All High", rows))
    per_case_fields = [
        "case_id", "response_state", "subgroup", "voxtell_empty", "voxtell_prediction_available",
        "sat_prediction_available", "sat_prediction_source", "voxtell_dice", "voxtell_precision",
        "voxtell_recall", "voxtell_foreground_voxels", "sat_dice", "sat_precision", "sat_recall",
        "sat_foreground_voxels", "gt_foreground_voxels", "intersection_dice", "intersection_precision",
        "intersection_recall", "union_dice", "union_precision", "union_recall", "sat_only_precision",
        "voxtell_only_precision", "overlap_metrics_available", "sat_minus_voxtell_dice",
        "best_of_two_dice", "best_of_two_uplift",
    ]
    with (ANALYSIS / "per_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_case_fields)
        writer.writeheader()
        writer.writerows(rows)
    with (ANALYSIS / "subgroup_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(subgroup_rows[0]))
        writer.writeheader()
        writer.writerows(subgroup_rows)

    manifest = json.loads((ANALYSIS / "input_manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "analysis_cases": len(rows),
        "voxtell_masks_available": sum(r["voxtell_prediction_available"] for r in rows),
        "voxtell_masks_missing": sum(not r["voxtell_prediction_available"] for r in rows),
        "sat_predictions_available": sum(r["sat_prediction_available"] for r in rows),
        "subgroups": {r["subgroup"]: r["n"] for r in subgroup_rows},
    })
    (ANALYSIS / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "groups": manifest["subgroups"],
                      "best_of_two_mean_uplift": subgroup_rows[-1]["best_of_two_mean_uplift"],
                      "voxtell_masks_available": manifest["voxtell_masks_available"]}, indent=2))


if __name__ == "__main__":
    main()
