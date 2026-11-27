from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from scipy.stats import pearsonr, spearmanr

from cue_sfda.data import TargetVolumeStore, load_split


def correlation(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_r": float("nan"), "spearman_p": float("nan")}
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def read_cache_case(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as payload:
        probability = payload["joint_probability"][0].astype(np.float32, copy=False)
        confidence = payload["joint_voxel_confidence"][0].astype(np.float32, copy=False)
        geometry = json.loads(str(payload["geometry_json"].item()))
    return probability, confidence, geometry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline GT audit of Dynamic-hard consensus before and after refresh"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=Path("/data/hilab/datasets/WCH_MRI145/DWI/images"))
    parser.add_argument("--split-json", type=Path, default=Path("/data/hilab/datasets/WCH_MRI145/DWI/split.json"))
    parser.add_argument("--gt-dir", type=Path, default=Path("/data/hilab/datasets/WCH_MRI145/DWI/labels"))
    parser.add_argument("--patch-size", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--tile-step-size", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_cases, _ = load_split(args.image_dir, args.split_json)
    store = TargetVolumeStore(tuple(args.patch_size), cache_size=1)
    reader = NibabelIOWithReorient()
    rows = []
    for case in train_cases:
        before_path = args.cache_root / "version_0000" / "cases" / f"{case.name}.npz"
        after_path = args.cache_root / "version_0001" / "cases" / f"{case.name}.npz"
        if not before_path.is_file() or not after_path.is_file():
            raise FileNotFoundError(f"Missing cache version for {case.name}")
        _, _, geometry = store.sliding_window_with_geometry(case, args.tile_step_size)
        roi_slicer = geometry["roi_slicer"]
        image, properties = reader.read_images([str(case.image_path)])
        gt, _ = reader.read_seg(str(args.gt_dir / case.name))
        if gt.ndim == 3:
            gt = gt[None]
        _, gt_crop, _ = crop_to_nonzero(image.astype(np.float32, copy=False), gt)
        ground_truth = np.asarray(gt_crop[0] == args.label_value)

        outputs = {}
        for version, path in ((0, before_path), (1, after_path)):
            probability, confidence, saved_geometry = read_cache_case(path)
            if tuple(probability.shape) != tuple(geometry["padded_shape"]):
                raise ValueError(f"Cache/case padded geometry mismatch for {case.name} version {version}")
            probability_roi = probability[roi_slicer]
            confidence_roi = confidence[roi_slicer]
            if probability_roi.shape != ground_truth.shape:
                raise ValueError(f"Cache/GT ROI geometry mismatch for {case.name}")
            prediction = probability_roi >= args.threshold
            intersection = int(np.logical_and(prediction, ground_truth).sum())
            denominator = int(prediction.sum()) + int(ground_truth.sum())
            dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
            union = int(np.logical_or(prediction, ground_truth).sum())
            iou = 1.0 if union == 0 else intersection / union
            outputs[version] = {
                "dice": float(dice),
                "iou": float(iou),
                "predicted_voxels": int(prediction.sum()),
                "empty_prediction": int(not bool(prediction.any())),
                "mean_voxel_confidence": float(confidence_roi.mean()),
                "original_prediction_empty": int(bool(saved_geometry.get("original_prediction_empty", False))),
            }
        rows.append(
            {
                "case_id": case.name,
                **{f"v0_{key}": value for key, value in outputs[0].items()},
                **{f"v1_{key}": value for key, value in outputs[1].items()},
                "dice_change": outputs[1]["dice"] - outputs[0]["dice"],
                "predicted_voxels_change": outputs[1]["predicted_voxels"] - outputs[0]["predicted_voxels"],
                "mean_voxel_confidence_change": outputs[1]["mean_voxel_confidence"] - outputs[0]["mean_voxel_confidence"],
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "refresh_consensus_per_case.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dice_changes = [float(row["dice_change"]) for row in rows]
    confidence_changes = [float(row["mean_voxel_confidence_change"]) for row in rows]
    summary = {
        "experiment": args.experiment,
        "num_cases": len(rows),
        "version0_mean_consensus_dice": float(np.mean([row["v0_dice"] for row in rows])),
        "version1_mean_consensus_dice": float(np.mean([row["v1_dice"] for row in rows])),
        "mean_consensus_dice_change": float(np.mean(dice_changes)),
        "version0_mean_predicted_voxels": float(np.mean([row["v0_predicted_voxels"] for row in rows])),
        "version1_mean_predicted_voxels": float(np.mean([row["v1_predicted_voxels"] for row in rows])),
        "mean_predicted_voxels_change": float(np.mean([row["predicted_voxels_change"] for row in rows])),
        "version0_empty_consensus_predictions": sum(int(row["v0_empty_prediction"]) for row in rows),
        "version1_empty_consensus_predictions": sum(int(row["v1_empty_prediction"]) for row in rows),
        "version0_mean_voxel_confidence": float(np.mean([row["v0_mean_voxel_confidence"] for row in rows])),
        "version1_mean_voxel_confidence": float(np.mean([row["v1_mean_voxel_confidence"] for row in rows])),
        "confidence_change_vs_dice_change": correlation(confidence_changes, dice_changes),
        "gt_usage": "offline cache-target diagnostic only; not used in training or cache construction",
    }
    (output_dir / "refresh_consensus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote Dynamic-hard refresh diagnostics for {len(rows)} cases to {output_dir}")


if __name__ == "__main__":
    main()
