#!/usr/bin/env python3
"""Offline SAT diagnostic on D-joint-gate No-response training cases.

SAT is run separately by the official SAT inference entrypoint.  This script
builds its JSONL input and computes metrics in SAT's own RAS/1x1x3/cropped
coordinate system, applying the identical transform to the GT.
"""
from __future__ import annotations

import argparse
import csv
import json
import importlib.machinery
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
if not hasattr(importlib.machinery.FileFinder, "find_module"):
    importlib.machinery.FileFinder.find_module = lambda self, fullname: self.find_spec(fullname).loader
import monai


def cases_from_routing(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r["case_id"] for r in csv.DictReader(f) if r["response_state"] == "No-response"]


def transform_gt(image: Path, label: Path) -> np.ndarray:
    # Match ../../SAT/runtime_compat/data/inference_dataset.py exactly, with nearest-neighbor labels.
    tx = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image", "label"]),
        monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
        monai.transforms.Orientationd(axcodes="RAS", keys=["image", "label"]),
        monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
        monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        monai.transforms.ToTensord(keys=["image", "label"]),
    ])
    out = tx({"image": str(image), "label": str(label)})
    return out["label"].numpy()[0] > 0


def metric(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    npred = int(pred.sum())
    ngt = int(gt.sum())
    dice = 1.0 if npred == 0 and ngt == 0 else (2.0 * inter / (npred + ngt) if npred + ngt else 0.0)
    precision = inter / npred if npred else 0.0
    recall = inter / ngt if ngt else 0.0
    return dice, precision, recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-csv", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--sat-results", type=Path, required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = cases_from_routing(args.routing_csv)
    rows_json = []
    for case in case_ids:
        image = args.image_dir / case
        if not image.is_file():
            raise FileNotFoundError(image)
        rows_json.append({"image": str(image), "label": ["liver cancer"], "dataset": "D_no_response", "modality": "mri", "orientation_code": "RAS"})
    with args.jsonl.open("w", encoding="utf-8") as f:
        for row in rows_json:
            f.write(json.dumps(row) + "\n")

    rows = []
    for case in case_ids:
        stem = case[:-7] if case.endswith(".nii.gz") else Path(case).stem
        pred_path = args.sat_results / "D_no_response" / f"seg_{stem}" / "liver cancer.nii.gz"
        # Official SAT also writes the aggregated volume; prefer the class mask.
        if not pred_path.is_file():
            pred_path = args.sat_results / "D_no_response" / f"seg_{stem}.nii.gz"
        if not pred_path.is_file():
            raise FileNotFoundError(f"SAT prediction missing for {case}: {pred_path}")
        pred = np.asarray(nib.load(str(pred_path)).dataobj) > 0.5
        gt = transform_gt(args.image_dir / case, args.gt_dir / case)
        if pred.shape != gt.shape:
            raise RuntimeError(f"Aligned shape mismatch {case}: SAT={pred.shape}, GT={gt.shape}")
        dice, precision, recall = metric(pred, gt)
        rows.append({
            "case_id": case,
            "voxtell_empty": True,
            "sat_nonempty": bool(pred.any()),
            "sat_prediction_voxels": int(pred.sum()),
            "gt_voxels": int(gt.sum()),
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "sat_prediction_path": str(pred_path),
        })
    df = pd.DataFrame(rows).sort_values("case_id")
    df.to_csv(args.output_dir / "per_case.csv", index=False)
    n = len(df)
    nonempty = int(df.sat_nonempty.sum())
    summary = {
        "prompt": "liver cancer",
        "n_no_response": n,
        "sat_nonempty_recovered": nonempty,
        "sat_nonempty_recovered_rate": nonempty / n if n else 0.0,
        "dice_gt_0.1_count": int((df.dice > 0.1).sum()),
        "dice_gt_0.1_rate": float((df.dice > 0.1).mean()) if n else 0.0,
        "dice_gt_0.3_count": int((df.dice > 0.3).sum()),
        "dice_gt_0.3_rate": float((df.dice > 0.3).mean()) if n else 0.0,
        "dice_gt_0.5_count": int((df.dice > 0.5).sum()),
        "dice_gt_0.5_rate": float((df.dice > 0.5).mean()) if n else 0.0,
        "sat_nonempty_dice_le_0.1_count": int((df.sat_nonempty & (df.dice <= 0.1)).sum()),
        "sat_nonempty_dice_le_0.1_rate_of_nonempty": float((df.loc[df.sat_nonempty, "dice"] <= 0.1).mean()) if nonempty else 0.0,
        "mean_dice": float(df.dice.mean()) if n else 0.0,
        "mean_precision": float(df.precision.mean()) if n else 0.0,
        "mean_recall": float(df.recall.mean()) if n else 0.0,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
