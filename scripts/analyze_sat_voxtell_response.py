#!/usr/bin/env python3
"""Measure frozen VoxTell soft response inside already-generated SAT masks."""
from __future__ import annotations

import argparse
import csv
import importlib.machinery
import json
from pathlib import Path

if not hasattr(importlib.machinery.FileFinder, "find_module"):
    importlib.machinery.FileFinder.find_module = lambda self, fullname: self.find_spec(fullname).loader

import matplotlib.pyplot as plt
import torch
try:
    _tvlib = torch.library.Library("torchvision", "DEF")
    _tvlib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
import numpy as np
import pandas as pd
import monai
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from cue_sfda.config import load_config
from cue_sfda.data import CaseRecord
from cue_sfda.evaluation import predict_volume
from cue_sfda.modeling import build_runtime
from cue_sfda.utils import seed_everything
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization


def sat_path(root: Path, case: str, dataset: str = "D_no_response") -> Path:
    stem = case[:-7] if case.endswith(".nii.gz") else Path(case).stem
    p = root / dataset / f"seg_{stem}" / "liver cancer.nii.gz"
    return p if p.is_file() else root / dataset / f"seg_{stem}.nii.gz"


def to_sat_grid(image: Path, probability: Path) -> np.ndarray:
    # Geometry is identical to SAT/data/inference_dataset.py. The source image
    # drives CropForegroundd so a near-zero probability map is never cropped.
    tx = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image", "prob"]),
        monai.transforms.EnsureChannelFirstd(keys=["image", "prob"]),
        monai.transforms.Orientationd(axcodes="RAS", keys=["image", "prob"]),
        monai.transforms.Spacingd(keys=["image", "prob"], pixdim=(1, 1, 3), mode=("bilinear", "bilinear")),
        monai.transforms.CropForegroundd(keys=["image", "prob"], source_key="image"),
        monai.transforms.ToTensord(keys=["image", "prob"]),
    ])
    return tx({"image": str(image), "prob": str(probability)})["prob"].numpy()[0]


def response_values(probability: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = probability[mask]
    if values.size == 0:
        return {"voxtell_mean_probability": 0.0, "voxtell_max_probability": 0.0,
                "voxtell_top1pct_mean_probability": 0.0, "voxtell_top5pct_mean_probability": 0.0}
    values = np.asarray(values, dtype=np.float64)
    return {
        "voxtell_mean_probability": float(values.mean()),
        "voxtell_max_probability": float(values.max()),
        "voxtell_top1pct_mean_probability": float(np.sort(values)[-max(1, int(np.ceil(values.size * .01))):].mean()),
        "voxtell_top5pct_mean_probability": float(np.sort(values)[-max(1, int(np.ceil(values.size * .05))):].mean()),
    }


def save_probability_reoriented(probability: np.ndarray, path: Path, properties: dict) -> None:
    # Float-preserving equivalent of NibabelIOWithReorient.write_seg.
    arr = probability.transpose((2, 1, 0)).astype(np.float32, copy=False)
    seg_nib = nib.Nifti1Image(arr, affine=properties["nibabel_stuff"]["reoriented_affine"])
    img_ornt = io_orientation(properties["nibabel_stuff"]["original_affine"])
    from_canonical = ornt_transform(axcodes2ornt("RAS"), img_ornt)
    nib.save(seg_nib.as_reoriented(from_canonical), str(path))


def metric(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    npred, ngt = int(pred.sum()), int(gt.sum())
    dice = 1.0 if npred == 0 and ngt == 0 else (2 * inter / (npred + ngt) if npred + ngt else 0.0)
    return dice, inter / npred if npred else 0.0, inter / ngt if ngt else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--routing-csv", type=Path, required=True)
    ap.add_argument("--sat-results", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--routing-dataset", default="D_no_response")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["experiment"]["gpu_id"] = args.gpu_id
    seed_everything(int(config["experiment"]["seed"]))
    with args.routing_csv.open(encoding="utf-8-sig", newline="") as f:
        cases = [r["case_id"] for r in csv.DictReader(f) if r["response_state"] == "No-response"]
    runtime = build_runtime(config, args.output_dir / "prompt_cache")
    runtime.model.eval()
    reader = NibabelIOWithReorient()
    normalizer = ZScoreNormalization(intensityproperties={})
    rows = []
    prob_dir = args.output_dir / "voxtell_probability"
    prob_dir.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases, 1):
        image_path = Path(config["data"]["image_dir"]) / case
        gt_path = args.gt_dir / case
        sat_file = sat_path(args.sat_results, case, args.routing_dataset)
        if not sat_file.is_file():
            raise FileNotFoundError(sat_file)
        image, props = reader.read_images([str(image_path)])
        probability = predict_volume(runtime, image, normalizer, config)
        prob_path = prob_dir / case
        # NibabelIOWithReorient.write_seg intentionally stores integer labels;
        # use a float NIfTI here so sub-0.5 soft probabilities are preserved.
        save_probability_reoriented(probability, prob_path, props)
        sat_mask = np.asarray(nib.load(str(sat_file)).dataobj) > .5
        probability_sat = to_sat_grid(image_path, prob_path)
        if probability_sat.shape != sat_mask.shape:
            raise RuntimeError(f"shape mismatch {case}: VoxTell={probability_sat.shape}, SAT={sat_mask.shape}")
        gt = np.asarray(nib.load(str(gt_path)).dataobj) > 0
        # GT is only for post-hoc labels/metrics; response values use SAT mask only.
        gt_sat = to_sat_grid(image_path, gt_path) > 0
        dice, precision, recall = metric(sat_mask, gt_sat)
        row = {"case_id": case, "sat_prediction_voxels": int(sat_mask.sum()), **response_values(probability_sat, sat_mask),
               "sat_dice": dice, "sat_precision": precision, "sat_recall": recall,
               "effective_proposal": bool(dice > .1), "voxtell_probability_path": str(prob_path)}
        rows.append(row)
        print(f"{i}/{len(cases)} {case} sat_voxels={row['sat_prediction_voxels']} top1={row['voxtell_top1pct_mean_probability']:.5f}", flush=True)
    df = pd.DataFrame(rows).sort_values("case_id")
    df.to_csv(args.output_dir / "per_case.csv", index=False)

    label = df.effective_proposal.astype(int).to_numpy()
    metrics = ["sat_prediction_voxels", "voxtell_mean_probability", "voxtell_max_probability",
               "voxtell_top1pct_mean_probability", "voxtell_top5pct_mean_probability"]
    correlation_rows, classifier_rows = [], []
    for name in metrics:
        x, y = df[name].to_numpy(float), df.sat_dice.to_numpy(float)
        correlation_rows.append({"metric": name, "pearson_r": float(pearsonr(x, y).statistic),
                                 "pearson_p": float(pearsonr(x, y).pvalue),
                                 "spearman_r": float(spearmanr(x, y).statistic),
                                 "spearman_p": float(spearmanr(x, y).pvalue)})
        classifier_rows.append({"metric": name, "auroc": float(roc_auc_score(label, x)) if len(np.unique(label)) == 2 else np.nan,
                                "auprc": float(average_precision_score(label, x)) if len(np.unique(label)) == 2 else np.nan})
    pd.DataFrame(correlation_rows).to_csv(args.output_dir / "correlations.csv", index=False)
    pd.DataFrame(classifier_rows).to_csv(args.output_dir / "classification_scores.csv", index=False)

    # Fixed diagnostic thresholds only; no threshold is fitted or fed back to training.
    sweep = []
    for name in ["voxtell_mean_probability", "voxtell_max_probability", "voxtell_top1pct_mean_probability", "voxtell_top5pct_mean_probability"]:
        for threshold in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
            selected = df[name] > threshold
            sweep.append({"metric": name, "threshold": threshold, "selected": int(selected.sum()),
                          "effective_selected": int((selected & df.effective_proposal).sum()),
                          "precision_vs_gt_label": float((selected & df.effective_proposal).sum() / selected.sum()) if selected.sum() else np.nan,
                          "recall_vs_gt_label": float((selected & df.effective_proposal).sum() / df.effective_proposal.sum()) if df.effective_proposal.sum() else np.nan})
    pd.DataFrame(sweep).to_csv(args.output_dir / "fixed_threshold_sweep.csv", index=False)

    for xname, outfile in [("voxtell_top1pct_mean_probability", "scatter_dice_vs_top1.png"), ("sat_prediction_voxels", "scatter_dice_vs_volume.png")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df[xname], df.sat_dice, c=df.effective_proposal.map({True: "tab:green", False: "tab:red"}))
        for _, r in df.iterrows():
            ax.annotate(r.case_id.replace(".nii.gz", ""), (r[xname], r.sat_dice), fontsize=7, alpha=.8)
        ax.set_xlabel(xname); ax.set_ylabel("SAT Dice"); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(args.output_dir / outfile, dpi=180); plt.close(fig)
    summary = {"n": len(df), "effective_proposals": int(df.effective_proposal.sum()), "ineffective_proposals": int((~df.effective_proposal).sum()),
               "mean_response_effective": df.loc[df.effective_proposal, "voxtell_top1pct_mean_probability"].mean(),
               "mean_response_ineffective": df.loc[~df.effective_proposal, "voxtell_top1pct_mean_probability"].mean(),
               "small_sample_warning": "n=17; correlation and classifier estimates are exploratory and unstable"}
    with (args.output_dir / "summary.json").open("w") as f: json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
