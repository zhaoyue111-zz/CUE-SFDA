from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from scipy.stats import pearsonr, spearmanr


def case_id(name: str) -> str:
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def read_csv_map(path: Path, key: str, value: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row[key]): float(row[value]) for row in csv.DictReader(handle)}


def corr(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_r": float("nan"), "spearman_p": float("nan")}
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    denominator = int(prediction.sum()) + int(ground_truth.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int(np.logical_and(prediction, ground_truth).sum()) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline, GT-only diagnostics for Dynamic-hard cache refreshes"
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--before-version", type=int, default=0)
    parser.add_argument("--after-version", type=int, default=1)
    parser.add_argument("--image-dir", type=Path, default=Path("/data/hilab/datasets/WCH_MRI145/DWI/images"))
    parser.add_argument("--gt-dir", type=Path, default=Path("/data/hilab/datasets/WCH_MRI145/DWI/labels"))
    parser.add_argument("--source-probability-dir", type=Path, default=Path("analysis/hierarchical_uncertainty/stage1_image_level/probabilities"))
    parser.add_argument("--source-pixel-weight-dir", type=Path, default=Path("analysis/hierarchical_dynamic/fixed_pixel_uncertainty"))
    parser.add_argument("--initial-confidence-csv", type=Path, default=Path("analysis/hierarchical_dynamic/image_confidence.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(192, 192, 192))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label-value", type=int, default=1)
    args = parser.parse_args()

    cache_root = args.cache_root.resolve()
    before_root = cache_root / f"version_{args.before_version:04d}" / "cases"
    after_root = cache_root / f"version_{args.after_version:04d}" / "cases"
    if not before_root.is_dir() or not after_root.is_dir():
        raise FileNotFoundError("Both Dynamic-hard cache versions must exist")
    confidence_before = read_csv_map(
        args.initial_confidence_csv, "case_id", "C_case"
    )
    reader = NibabelIOWithReorient()
    rows = []
    for after_path in sorted(after_root.glob("*.nii.gz.npz")):
        name = after_path.name[:-4]
        cid = case_id(name)
        before_path = before_root / after_path.name
        source_path = args.source_probability_dir / f"{cid}.npz"
        source_weight_path = args.source_pixel_weight_dir / f"{cid}.npz"
        image_path = args.image_dir / name
        gt_path = args.gt_dir / name
        if not before_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"Missing before/source cache for {name}")
        image, properties = reader.read_images([str(image_path)])
        ground_truth, _ = reader.read_seg(str(gt_path))
        if ground_truth.ndim == 3:
            ground_truth = ground_truth[None]
        _, cropped_gt, _ = crop_to_nonzero(
            image.astype(np.float32, copy=False), ground_truth
        )
        cropped_gt = np.asarray(cropped_gt[0] == args.label_value)

        with np.load(source_path, allow_pickle=False) as source:
            source_probability = source["original"].astype(np.float32, copy=False)
        source_prediction = np.squeeze(source_probability) >= args.threshold
        if source_prediction.shape != cropped_gt.shape:
            raise ValueError(
                f"Source/GT cropped geometry mismatch for {name}: "
                f"{source_prediction.shape} vs {cropped_gt.shape}"
            )

        with np.load(after_path, allow_pickle=False) as after:
            geometry = json.loads(str(after["geometry_json"].item()))
            if "quality_prediction_mask" not in after.files:
                raise ValueError(
                    f"Refreshed cache has no benchmark quality mask: {after_path}"
                )
            padded_prediction = after["quality_prediction_mask"][0].astype(bool)
            refreshed_pixel_weight = after["pixel_weight"][0].astype(np.float32)
        roi_slicer = tuple(slice(*axis) for axis in geometry["roi_slicer"])
        refreshed_prediction = padded_prediction[roi_slicer]
        refreshed_weight_roi = refreshed_pixel_weight[roi_slicer]
        with np.load(source_weight_path, allow_pickle=False) as source_weight:
            before_weight = source_weight["weight"][0].astype(np.float32)
            before_metadata = json.loads(str(source_weight["metadata"].item()))
        before_roi_slicer = tuple(
            slice(*axis) for axis in before_metadata["roi_slicer"]
        )
        source_weight_roi = before_weight[before_roi_slicer]
        if refreshed_prediction.shape != cropped_gt.shape:
            raise ValueError(
                f"Refreshed/GT cropped geometry mismatch for {name}: "
                f"{refreshed_prediction.shape} vs {cropped_gt.shape}"
            )
        if source_weight_roi.shape != cropped_gt.shape or refreshed_weight_roi.shape != cropped_gt.shape:
            raise ValueError(
                f"Pixel-weight/GT cropped geometry mismatch for {name}: "
                f"{source_weight_roi.shape}, {refreshed_weight_roi.shape}, {cropped_gt.shape}"
            )
        confidence_after = float(geometry["case_confidence"])
        confidence_initial = float(confidence_before[cid])
        source_empty = not bool(source_prediction.any())
        refreshed_empty = not bool(refreshed_prediction.any())
        rows.append(
            {
                "case_id": cid,
                "initial_confidence": confidence_initial,
                "refreshed_confidence": confidence_after,
                "confidence_change": confidence_after - confidence_initial,
                "initial_case_uncertainty": 1.0 - confidence_initial,
                "refreshed_case_uncertainty": 1.0 - confidence_after,
                "case_uncertainty_change": confidence_initial - confidence_after,
                "mean_pixel_weight_initial": float(source_weight_roi.mean()),
                "mean_pixel_weight_refreshed": float(refreshed_weight_roi.mean()),
                "mean_pixel_weight_change": float(refreshed_weight_roi.mean() - source_weight_roi.mean()),
                "mean_pixel_uncertainty_initial": float(1.0 - source_weight_roi.mean()),
                "mean_pixel_uncertainty_refreshed": float(1.0 - refreshed_weight_roi.mean()),
                "mean_pixel_uncertainty_change": float(source_weight_roi.mean() - refreshed_weight_roi.mean()),
                "source_dice": dice(source_prediction, cropped_gt),
                "refreshed_dice": dice(refreshed_prediction, cropped_gt),
                "dice_change": dice(refreshed_prediction, cropped_gt) - dice(source_prediction, cropped_gt),
                "source_predicted_voxels": int(source_prediction.sum()),
                "refreshed_predicted_voxels": int(refreshed_prediction.sum()),
                "predicted_voxels_change": int(refreshed_prediction.sum()) - int(source_prediction.sum()),
                "source_empty": int(source_empty),
                "refreshed_empty": int(refreshed_empty),
                "new_empty_prediction": int(not source_empty and refreshed_empty),
                "recovered_from_empty": int(source_empty and not refreshed_empty),
            }
        )

    if len(rows) != len(confidence_before):
        raise ValueError(
            f"Expected {len(confidence_before)} refreshed cases, found {len(rows)}"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "refresh_case_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    confidence_delta = [float(row["confidence_change"]) for row in rows]
    dice_delta = [float(row["dice_change"]) for row in rows]
    summary = {
        "before_version": args.before_version,
        "after_version": args.after_version,
        "num_cases": len(rows),
        "mean_initial_confidence": float(np.mean([row["initial_confidence"] for row in rows])),
        "mean_refreshed_confidence": float(np.mean([row["refreshed_confidence"] for row in rows])),
        "mean_confidence_change": float(np.mean(confidence_delta)),
        "mean_source_dice": float(np.mean([row["source_dice"] for row in rows])),
        "mean_refreshed_dice": float(np.mean([row["refreshed_dice"] for row in rows])),
        "mean_dice_change": float(np.mean(dice_delta)),
        "source_empty_predictions": sum(int(row["source_empty"]) for row in rows),
        "refreshed_empty_predictions": sum(int(row["refreshed_empty"]) for row in rows),
        "new_empty_predictions": sum(int(row["new_empty_prediction"]) for row in rows),
        "recovered_from_empty": sum(int(row["recovered_from_empty"]) for row in rows),
        "confidence_change_vs_dice_change": corr(confidence_delta, dice_delta),
        "gt_role": "offline diagnostics only; never used in training, sampling, confidence, or cache construction",
    }
    (output / "refresh_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote refresh diagnostics for {len(rows)} cases to {output}")


if __name__ == "__main__":
    main()
