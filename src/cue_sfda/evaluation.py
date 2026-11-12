from __future__ import annotations

import csv
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.sliding_window_prediction import (
    compute_steps_for_sliding_window,
)
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    ZScoreNormalization,
)
from cue_sfda.data import CaseRecord
from cue_sfda.modeling.voxtell import VoxTellRuntime


def evaluate(
    config: dict,
    runtime: VoxTellRuntime,
    cases: list[CaseRecord],
    gt_path: str | Path,
    output_dir: str | Path,
    logger,
    save_predictions: bool = True,
) -> dict[str, float]:
    gt_root = Path(gt_path).expanduser().resolve()
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT directory does not exist: {gt_root}")
    output_root = Path(output_dir)
    prediction_dir = output_root / "predictions"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_root.mkdir(parents=True, exist_ok=True)
    reader = NibabelIOWithReorient()
    normalizer = ZScoreNormalization(intensityproperties={})
    rows = []
    runtime.model.eval()
    for index, case in enumerate(cases, start=1):
        gt_file = gt_root / case.name
        if not gt_file.is_file():
            raise FileNotFoundError(f"Missing GT file: {gt_file}")
        image, properties = reader.read_images([str(case.image_path)])
        label, _ = reader.read_seg(str(gt_file))
        probability = predict_volume(runtime, image, normalizer, config)
        threshold = float(config["evaluation"]["threshold"])
        prediction = probability >= threshold
        label_value = int(config["evaluation"]["label_value"])
        ground_truth = label == label_value
        dice, iou = binary_metrics(prediction, ground_truth)
        if save_predictions:
            prediction_path = prediction_dir / case.name
            reader.write_seg(
                prediction.astype(np.uint8),
                str(prediction_path),
                properties,
            )
        rows.append(
            {
                "case": case.name,
                "dice": dice,
                "iou": iou,
                "predicted_voxels": int(prediction.sum()),
                "ground_truth_voxels": int(ground_truth.sum()),
            }
        )
        logger.info(
            "evaluation=%d/%d case=%s dice=%.6f iou=%.6f",
            index,
            len(cases),
            case.name,
            dice,
            iou,
        )
    metrics_path = output_root / "metrics_per_case.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "num_cases": len(rows),
    }
    summary_path = output_root / "metrics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    logger.info(
        "Evaluation complete: mean_dice=%.6f mean_iou=%.6f n=%d",
        summary["mean_dice"],
        summary["mean_iou"],
        summary["num_cases"],
    )
    return summary


@torch.inference_mode()
def predict_volume(
    runtime: VoxTellRuntime,
    image: np.ndarray,
    normalizer: ZScoreNormalization,
    config: dict,
) -> np.ndarray:
    data = image.astype(np.float32, copy=True)
    original_shape = data.shape[1:]
    data, _, bounding_box = crop_to_nonzero(data, None)
    data = normalizer.run(data, None)
    data_tensor = torch.from_numpy(np.ascontiguousarray(data)).float()
    patch_size = tuple(int(value) for value in config["data"]["patch_size"])
    padded, revert_slicer = pad_nd_image(
        data_tensor,
        patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )
    spatial_shape = tuple(int(value) for value in padded.shape[1:])
    step_size = float(config["evaluation"]["tile_step_size"])
    steps = compute_steps_for_sliding_window(spatial_shape, patch_size, step_size)
    amp_enabled = bool(config["model"].get("amp", True))
    amp_enabled = amp_enabled and runtime.device.type == "cuda"
    accumulation_dtype = torch.float16 if amp_enabled else torch.float32
    accumulated_logits = torch.zeros(
        spatial_shape,
        dtype=accumulation_dtype,
        device=runtime.device,
    )
    normalization = torch.zeros(
        spatial_shape,
        dtype=accumulation_dtype,
        device=runtime.device,
    )
    gaussian = compute_gaussian(
        patch_size,
        sigma_scale=1.0 / 8.0,
        value_scaling_factor=10,
        dtype=accumulation_dtype,
        device=runtime.device,
    )
    amp_context = torch.autocast if amp_enabled else None
    for start_d in steps[0]:
        for start_h in steps[1]:
            for start_w in steps[2]:
                spatial_slice = (
                    slice(start_d, start_d + patch_size[0]),
                    slice(start_h, start_h + patch_size[1]),
                    slice(start_w, start_w + patch_size[2]),
                )
                tile = padded[(slice(None), *spatial_slice)].unsqueeze(0)
                tile = tile.to(runtime.device, non_blocking=True)
                context = (
                    amp_context("cuda", dtype=torch.float16)
                    if amp_context
                    else nullcontext()
                )
                with context:
                    logits = runtime.model(tile, runtime.embedding)
                tile_logits = logits[0, 0].to(dtype=accumulation_dtype)
                accumulated_logits[spatial_slice] += tile_logits * gaussian
                normalization[spatial_slice] += gaussian
    fused_logits = accumulated_logits / normalization.clamp_min(1e-6)
    cropped_probability = torch.sigmoid(fused_logits.float())
    cropped_probability = cropped_probability[revert_slicer[1:]].cpu().numpy()
    restored = np.zeros(original_shape, dtype=np.float32)
    restored = insert_crop_into_image(restored, cropped_probability, bounding_box)
    return restored


def binary_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    epsilon: float = 1e-7,
) -> tuple[float, float]:
    prediction = prediction.astype(bool, copy=False)
    ground_truth = ground_truth.astype(bool, copy=False)
    prediction_sum = int(prediction.sum())
    ground_truth_sum = int(ground_truth.sum())
    if ground_truth_sum == 0:
        score = 1.0 if prediction_sum == 0 else 0.0
        return score, score
    intersection = int(np.logical_and(prediction, ground_truth).sum())
    union = int(np.logical_or(prediction, ground_truth).sum())
    dice = (2.0 * intersection + epsilon) / (
        prediction_sum + ground_truth_sum + epsilon
    )
    iou = (intersection + epsilon) / (union + epsilon)
    return float(dice), float(iou)
