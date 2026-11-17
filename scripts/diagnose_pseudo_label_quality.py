#!/usr/bin/env python3
"""Offline quality-metric diagnostic on one frozen, aligned prediction set.

No optimizer or adaptation code is constructed.  The only use of GT labels is
after all predictions and unlabeled scores have been computed, for Dice and
case-selection diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml


PROMPT_NAMES = ("liver_cancer", "liver_tumor", "hcc", "hepatocellular_carcinoma")
METRICS = tuple(f"M{i}" for i in range(7))
TARGETS = ("original_dice", "joint_dice", "all_view_dice_mean", "all_view_dice_variance")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_csv_by_case(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            case_id = Path(row["case_id"]).name
            if case_id.endswith(".nii.gz"):
                case_id = case_id[:-7]
            else:
                case_id = Path(case_id).stem
            if case_id in rows:
                raise ValueError(f"duplicate case in source uncertainty CSV: {case_id}")
            rows[case_id] = row
    return rows


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def image_case_rng(seed: int, case_id: str) -> np.random.Generator:
    bits = int.from_bytes(hashlib.sha256(case_id.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(np.uint64(seed) ^ np.uint64(bits))


def bias_field(shape: tuple[int, int, int], rng: np.random.Generator, amplitude: float) -> np.ndarray:
    field = np.zeros(shape, dtype=np.float32)
    for axis, size in enumerate(shape):
        coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        shape_axis = [1, 1, 1]
        shape_axis[axis] = size
        coordinates = coordinates.reshape(shape_axis)
        linear, quadratic = rng.normal(0.0, 0.35, size=2)
        field += linear * coordinates + quadratic * (coordinates * coordinates - 1.0 / 3.0)
    maximum = float(np.max(np.abs(field)))
    if maximum > 0:
        field *= amplitude / maximum
    return 1.0 + field


def weak_augment(preprocessed: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Match the saved DWI uncertainty protocol; only intensity changes are used."""
    values = preprocessed.detach().cpu().numpy().astype(np.float32, copy=True)
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError(f"expected (1,D,H,W), got {values.shape}")
    scale = rng.uniform(0.95, 1.05)
    offset = rng.uniform(-0.05, 0.05)
    contrast = rng.uniform(0.90, 1.10)
    gamma = rng.uniform(0.90, 1.10)
    spatial = values[0]
    mean = float(np.mean(spatial, dtype=np.float64))
    spatial = (spatial - mean) * contrast + mean
    spatial = np.sign(spatial) * np.power(np.abs(spatial), gamma)
    spatial = (spatial * scale + offset) * bias_field(spatial.shape, rng, 0.05)
    spatial += rng.normal(0.0, rng.uniform(0.0, 0.03), size=spatial.shape).astype(np.float32)
    return torch.from_numpy(spatial[None].astype(np.float32, copy=False))


def dice_masks(first: np.ndarray, second: np.ndarray, both_empty_zero: bool = False) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    n1 = int(first.sum())
    n2 = int(second.sum())
    if n1 + n2 == 0:
        return 0.0 if both_empty_zero else 1.0
    return 2.0 * int(np.count_nonzero(first & second)) / (n1 + n2)


def dice_to_gt(pred: np.ndarray, gt_crop: np.ndarray, gt_full_count: int) -> float:
    pred = np.asarray(pred, dtype=bool)
    denom = int(pred.sum()) + int(gt_full_count)
    if denom == 0:
        return 1.0
    return 2.0 * int(np.count_nonzero(pred & gt_crop)) / denom


def soft_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(a @ a + b @ b)
    if denominator <= 1e-8:
        return 1.0
    return float(2.0 * (a @ b) / (denominator + 1e-8))


def mean_pairwise_soft_dice(predictions: np.ndarray) -> float:
    if predictions.shape[0] < 2:
        raise ValueError("at least two views required")
    values = [soft_dice(predictions[i], predictions[j])
              for i in range(predictions.shape[0])
              for j in range(i + 1, predictions.shape[0])]
    return float(np.mean(values, dtype=np.float64))


def bbox_slices(bbox: Any, shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    if len(bbox) != 3:
        raise ValueError(f"invalid crop bbox: {bbox}")
    slices = []
    for (start, stop), extent in zip(bbox, shape):
        start, stop = int(start), int(stop)
        if start < 0 or stop > extent or stop <= start:
            raise ValueError(f"invalid bbox interval {(start, stop)} for {extent}")
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, samples: np.ndarray) -> tuple[float, float, int]:
    from scipy.stats import rankdata
    valid_cases = np.isfinite(x) & np.isfinite(y)
    indices = samples[:, valid_cases]
    sx = x[valid_cases][indices]
    sy = y[valid_cases][indices]
    rx = rankdata(sx, axis=1)
    ry = rankdata(sy, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", rx, ry)
    denominator = np.sqrt(np.einsum("ij,ij->i", rx, rx) * np.einsum("ij,ij->i", ry, ry))
    values = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)), int(finite.size)


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    from scipy.stats import rankdata
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if valid.sum() < 4:
        return float("nan")
    rx, ry, rc = (rankdata(v[valid]).astype(np.float64) for v in (x, y, control))
    design = np.column_stack([np.ones(len(rc)), rc])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pseudo_label_quality_diagnostic.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/pseudo_label_quality_diagnostic"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug only; formal run omits this")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = args.output_dir.resolve()
    cache_dir = out / "view_prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device or config["model"]["device"]

    repo_root = Path.cwd().resolve()
    voxtell_root = Path(config["model"]["voxtell_root"]).resolve()
    sys.path.insert(0, str(voxtell_root))
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from cue_sfda.method.consensus import _consensus
    from voxtell.inference.predictor import VoxTellPredictor

    images = Path(config["data"]["image_dir"])
    labels = Path(config["data"]["labels_dir"])
    with Path(config["data"]["split_json"]).open("r", encoding="utf-8") as f:
        split = json.load(f)
    entries = split[config["data"]["split_key"]]
    cases = [Path(entry).name for entry in entries]
    if args.limit is not None:
        cases = cases[:args.limit]
    if len(cases) != int(config["data"]["expected_cases"]) and args.limit is None:
        raise RuntimeError(f"expected {config['data']['expected_cases']} training cases, found {len(cases)}")
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate case IDs in training split")
    for case in cases:
        if not (images / case).is_file() or not (labels / case).is_file():
            raise FileNotFoundError(f"missing image/label for {case}")
    source_csv = read_csv_by_case(Path(config["source_uncertainty_csv"]))
    case_stems = {Path(case).name[:-7] for case in cases}
    missing_source = sorted(case_stems - set(source_csv))
    extra_source = sorted(set(source_csv) - case_stems)
    if missing_source or (extra_source and args.limit is None):
        raise ValueError(f"source uncertainty CSV case mismatch: missing={missing_source[:5]}, extra={extra_source[:5]}")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {device}")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config["model"]["text_dtype"]]
    predictor = VoxTellPredictor(
        model_dir=str(Path(config["model"]["model_dir"])),
        device=device,
        text_encoding_model=str(Path(config["model"]["text_encoder"])),
        text_device=torch.device(config["model"]["text_device"]),
        text_dtype=dtype,
    )
    predictor.tile_step_size = float(config["inference"]["tile_step_size"])
    predictor.network = predictor.network.to(device).eval()
    predictor.network.requires_grad_(False)
    if predictor.tile_step_size != 0.5 or tuple(predictor.patch_size) != tuple(config["inference"]["patch_size"]):
        raise RuntimeError(f"inference protocol mismatch: patch={predictor.patch_size} step={predictor.tile_step_size}")
    embeddings = predictor.embed_text_prompts(list(config["model"]["prompts"]))
    reader = NibabelIOWithReorient()
    threshold = float(config["inference"]["probability_threshold"])
    image_view_count = int(config["augmentation"]["image_views_including_identity"])
    bootstrap_replicates = int(config["metrics"]["bootstrap_replicates"])
    rng_boot = np.random.default_rng(int(config["metrics"]["bootstrap_seed"]))
    bootstrap_indices = rng_boot.integers(0, len(cases), size=(bootstrap_replicates, len(cases)))
    rows: list[dict[str, Any]] = []
    view_dice_rows: list[dict[str, Any]] = []
    cache_manifest_cases: list[dict[str, Any]] = []
    csv_summary_rows: list[dict[str, Any]] = []

    for case_index, case_name in enumerate(cases, 1):
        case_id = case_name[:-7] if case_name.endswith(".nii.gz") else Path(case_name).stem
        image_path, label_path = images / case_name, labels / case_name
        case_cache_path = cache_dir / f"{case_id}.npz"
        image, properties = reader.read_images([str(image_path)])
        preprocessed, bbox, original_shape = predictor.preprocess(image)
        crop = bbox_slices(bbox, tuple(int(x) for x in original_shape))
        if tuple(preprocessed.shape[1:]) != tuple(c.stop - c.start for c in crop):
            raise ValueError(f"{case_id}: preprocessed image and crop bbox are misaligned")
        original_probability = predictor.predict_sliding_window_return_logits(preprocessed, embeddings[:, :1])
        if isinstance(original_probability, (tuple, list)):
            original_probability = original_probability[0]
        original_probability = torch.sigmoid(original_probability.float()).cpu().numpy()[0].astype(np.float32)
        prompt_logits = predictor.predict_sliding_window_return_logits(preprocessed, embeddings)
        if isinstance(prompt_logits, (tuple, list)):
            prompt_logits = prompt_logits[0]
        prompt_probabilities = torch.sigmoid(prompt_logits.float()).cpu().numpy().astype(np.float32)
        if prompt_probabilities.shape[0] != len(config["model"]["prompts"]):
            raise RuntimeError(f"{case_id}: expected 4 prompt outputs, got {prompt_probabilities.shape}")
        image_probabilities = [original_probability]
        rng_case = image_case_rng(int(config["augmentation"]["seed"]), case_id)
        for _ in range(1, image_view_count):
            augmented = weak_augment(preprocessed, rng_case)
            logits = predictor.predict_sliding_window_return_logits(augmented, embeddings[:, :1])
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            image_probabilities.append(torch.sigmoid(logits.float()).cpu().numpy()[0].astype(np.float32))
        image_probs = np.stack(image_probabilities)
        prompt_probs = prompt_probabilities
        expected_shape = tuple(preprocessed.shape[1:])
        if image_probs.shape[1:] != expected_shape or prompt_probs.shape[1:] != expected_shape:
            raise ValueError(f"{case_id}: view prediction shape mismatch; image={image_probs.shape}, prompt={prompt_probs.shape}, crop={expected_shape}")
        if not np.isfinite(image_probs).all() or not np.isfinite(prompt_probs).all():
            raise ValueError(f"{case_id}: nonfinite probability in prediction cache")

        image_consensus, _, image_weights = _consensus(image_probs[:, None], 0.15)
        prompt_consensus, _, prompt_weights = _consensus(prompt_probs[:, None], 0.15)
        image_fused = image_consensus[0]
        prompt_fused = prompt_consensus[0]
        joint_fused = (0.5 * image_fused + 0.5 * prompt_fused).astype(np.float32)

        # GT is deliberately loaded only after all unlabeled metrics were computed.
        gt_full = np.asarray(reader.read_images([str(label_path)])[0][0]) > 0
        if tuple(gt_full.shape) != tuple(original_shape):
            raise ValueError(f"{case_id}: GT/image space mismatch {gt_full.shape} vs {original_shape}")
        gt_crop = gt_full[crop]
        gt_count = int(gt_full.sum())
        if gt_crop.shape != expected_shape:
            raise ValueError(f"{case_id}: cropped GT and predictions mismatch")

        image_binary = image_probs >= threshold
        prompt_binary = prompt_probs >= threshold
        raw_binary = original_probability >= threshold
        # The original CUE-SFDA Chebyshev path uses a strict raw foreground
        # comparison (`raw_probability > threshold`). Keep that exact rule for
        # M2 while the other diagnostic masks use >= 0.5.
        raw_filter_binary = original_probability > threshold
        image_fused_binary = image_fused >= threshold
        prompt_fused_binary = prompt_fused >= threshold
        joint_binary = joint_fused >= threshold
        image_ui = 1.0 - mean_pairwise_soft_dice(image_probs)
        prompt_ut = 1.0 - mean_pairwise_soft_dice(prompt_probs)
        m0 = 1.0 - (image_ui + prompt_ut) / 2.0
        m1 = 0.0 if not raw_binary.any() else m0

        # Match the repository's Chebyshev filter: view-wise population variance,
        # q = var / (var + (mean - 0.5)^2 + eps), retain raw foreground where q <= .8.
        view_mean = image_probs.mean(axis=0, dtype=np.float64).astype(np.float32)
        view_variance = image_probs.var(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
        error_bound = view_variance / (view_variance + (view_mean - threshold) ** 2 + 1e-8)
        filtered_binary = raw_filter_binary & (error_bound <= float(config["metrics"]["chebyshev_bound_threshold"]))
        m2 = dice_masks(raw_filter_binary, filtered_binary, both_empty_zero=True)
        m3 = dice_masks(image_fused_binary, prompt_fused_binary, both_empty_zero=True)
        all_view_binary = [*image_binary, *prompt_binary]
        view_to_joint = [dice_masks(view, joint_binary, both_empty_zero=True) for view in all_view_binary]
        m4 = float(np.mean(view_to_joint, dtype=np.float64))
        metric_scores = {"M0": float(m0), "M1": float(m1), "M2": float(m2), "M3": float(m3),
                         "M4": float(m4), "M5": float(m0 * m2), "M6": float(m0 * m3)}

        raw_dice = dice_to_gt(raw_binary, gt_crop, gt_count)
        image_fused_dice = dice_to_gt(image_fused_binary, gt_crop, gt_count)
        prompt_fused_dice = dice_to_gt(prompt_fused_binary, gt_crop, gt_count)
        joint_dice = dice_to_gt(joint_binary, gt_crop, gt_count)
        image_dices = [dice_to_gt(mask, gt_crop, gt_count) for mask in image_binary]
        prompt_dices = [dice_to_gt(mask, gt_crop, gt_count) for mask in prompt_binary]
        all_dices = image_dices + prompt_dices
        all_view_mean = float(np.mean(all_dices, dtype=np.float64))
        all_view_var = float(np.var(all_dices, ddof=0, dtype=np.float64))

        source_row = source_csv.get(case_id)
        source_ui = float(source_row["image_uncertainty"]) if source_row and source_row.get("image_uncertainty", "") else float("nan")
        source_ut = float(source_row["prompt_uncertainty"]) if source_row and source_row.get("prompt_uncertainty", "") else float("nan")
        source_status = "valid" if np.isfinite(source_ui) and np.isfinite(source_ut) else "missing_or_invalid"
        row: dict[str, Any] = {
            "case_id": case_name,
            "original_dice": raw_dice,
            "image_fused_dice": image_fused_dice,
            "prompt_fused_dice": prompt_fused_dice,
            "joint_dice": joint_dice,
            "image_fused_foreground_voxels": int(image_fused_binary.sum()),
            "prompt_fused_foreground_voxels": int(prompt_fused_binary.sum()),
            "joint_foreground_voxels": int(joint_binary.sum()),
            "original_foreground_voxels": int(raw_binary.sum()),
            "gt_foreground_voxels": gt_count,
            "gt_crop_foreground_voxels": int(gt_crop.sum()),
            "all_view_dice_mean": all_view_mean,
            "all_view_dice_variance": all_view_var,
            "image_view_dice_mean": float(np.mean(image_dices)),
            "image_view_dice_std": float(np.std(image_dices, ddof=0)),
            "image_view_dice_variance": float(np.var(image_dices, ddof=0)),
            "prompt_view_dice_mean": float(np.mean(prompt_dices)),
            "prompt_view_dice_std": float(np.std(prompt_dices, ddof=0)),
            "prompt_view_dice_variance": float(np.var(prompt_dices, ddof=0)),
            "all_view_dice_std": float(np.std(all_dices, ddof=0)),
            "m0_image_uncertainty": image_ui,
            "m0_prompt_uncertainty": prompt_ut,
            "source_csv_image_uncertainty": source_ui,
            "source_csv_prompt_uncertainty": source_ut,
            "source_csv_join_status": source_status,
            "source_csv_M0_abs_difference": abs(m0 - (1.0 - (source_ui + source_ut) / 2.0)) if source_status == "valid" else float("nan"),
            "m2_filtered_foreground_voxels": int(filtered_binary.sum()),
            "m2_chebyshev_reliable_voxels": int(np.count_nonzero(error_bound <= float(config["metrics"]["chebyshev_bound_threshold"]))),
            "m3_both_fused_empty": int(not image_fused_binary.any() and not prompt_fused_binary.any()),
            "m4_mean_view_to_joint_dice": m4,
            "m4_empty_view_count": int(sum(not mask.any() for mask in all_view_binary)),
            "empty_original_prediction": int(not raw_binary.any()),
            "empty_image_fused_prediction": int(not image_fused_binary.any()),
            "empty_prompt_fused_prediction": int(not prompt_fused_binary.any()),
            "empty_joint_prediction": int(not joint_binary.any()),
            "m0_stratum": "",
            **{f"{k}_score": v for k, v in metric_scores.items()},
        }
        for idx, (view_dice, view_mask) in enumerate(zip(image_dices, image_binary)):
            row[f"image_view_{idx:02d}_dice"] = view_dice
            row[f"image_view_{idx:02d}_foreground_voxels"] = int(view_mask.sum())
            row[f"image_view_{idx:02d}_to_joint_dice"] = view_to_joint[idx]
            view_dice_rows.append({"case_id": case_name, "view_family": "image", "view_index": idx,
                                   "view_name": "identity" if idx == 0 else f"weak_aug_{idx:02d}",
                                   "dice": view_dice, "foreground_voxels": int(view_mask.sum()),
                                   "dice_to_joint": view_to_joint[idx]})
        for idx, (prompt, view_dice, view_mask) in enumerate(zip(config["model"]["prompts"], prompt_dices, prompt_binary)):
            row[f"prompt_view_{PROMPT_NAMES[idx]}_dice"] = view_dice
            row[f"prompt_view_{PROMPT_NAMES[idx]}_foreground_voxels"] = int(view_mask.sum())
            row[f"prompt_view_{PROMPT_NAMES[idx]}_to_joint_dice"] = view_to_joint[len(image_dices) + idx]
            view_dice_rows.append({"case_id": case_name, "view_family": "prompt", "view_index": idx,
                                   "view_name": str(prompt), "dice": view_dice,
                                   "foreground_voxels": int(view_mask.sum()),
                                   "dice_to_joint": view_to_joint[len(image_dices) + idx]})
        rows.append(row)

        # Persist the independent crop-space predictions, not fused substitutes.
        cache_payload = {
            "image_view_probabilities": image_probs.astype(np.float32, copy=False),
            "prompt_probabilities": prompt_probs.astype(np.float32, copy=False),
            "image_case_weights": image_weights.astype(np.float32, copy=False),
            "prompt_case_weights": prompt_weights.astype(np.float32, copy=False),
            "geometry_json": np.asarray(json.dumps({
                "case_id": case_name, "image_path": str(image_path.resolve()),
                "original_shape": list(original_shape),
                "crop_bbox": [[int(c.start), int(c.stop)] for c in crop],
                "crop_shape": list(expected_shape),
                "original_affine": np.asarray(properties.get("nibabel_stuff", {}).get("original_affine", np.eye(4))).tolist(),
                "reoriented_affine": np.asarray(properties.get("nibabel_stuff", {}).get("reoriented_affine", np.eye(4))).tolist(),
                "augmentation_geometry": "identity",
                "probabilities_in_original_eval_space_after_inverse_geometry": True,
                "threshold": threshold, "patch_size": list(config["inference"]["patch_size"]),
                "tile_step_size": predictor.tile_step_size, "gaussian_fusion": "logit-space; sigma_scale=1/8",
                "input_size": image_path.stat().st_size, "input_mtime_ns": image_path.stat().st_mtime_ns,
            }, sort_keys=True)),
        }
        with case_cache_path.open("wb") as f:
            np.savez_compressed(f, **cache_payload)
        cache_manifest_cases.append({"case_id": case_name, "cache": str(case_cache_path.relative_to(out)),
                                    "shape": list(expected_shape), "image_views": len(image_probs),
                                    "prompt_views": len(prompt_probs), "probability_dtype": "float32"})
        csv_summary_rows.append({"case_id": case_name, "recomputed_image_uncertainty": image_ui,
                                 "source_csv_image_uncertainty": source_ui,
                                 "recomputed_prompt_uncertainty": prompt_ut,
                                 "source_csv_prompt_uncertainty": source_ut,
                                 "recomputed_M0": m0,
                                 "source_csv_M0": 1.0 - (source_ui + source_ut) / 2.0 if source_status == "valid" else float("nan"),
                                 "max_abs_component_difference": max(abs(image_ui-source_ui), abs(prompt_ut-source_ut)) if source_status == "valid" else float("nan"),
                                 "status": source_status})
        print(f"[{case_index}/{len(cases)}] {case_name} M0={m0:.4f} original_Dice={raw_dice:.4f} joint_Dice={joint_dice:.4f}", flush=True)

    # M0 quintiles are a diagnostic control only; no thresholds feed prediction or scores.
    m0_values = np.asarray([row["M0_score"] for row in rows], dtype=np.float64)
    if len(rows) >= int(config["metrics"]["m0_strata"]):
        strata = np.array_split(np.argsort(m0_values, kind="mergesort"), int(config["metrics"]["m0_strata"]))
        for stratum_no, indices in enumerate(strata, 1):
            for idx in indices:
                rows[int(idx)]["m0_stratum"] = f"Q{stratum_no}"

    per_case_path = out / "per_case_metrics.csv"
    write_csv(per_case_path, rows)
    write_csv(out / "per_view_dice.csv", view_dice_rows)
    write_csv(out / "source_uncertainty_reproducibility.csv", csv_summary_rows)
    view_groups = {
        "image_views": [r["dice"] for r in view_dice_rows if r["view_family"] == "image"],
        "prompt_views": [r["dice"] for r in view_dice_rows if r["view_family"] == "prompt"],
        "all_views": [r["dice"] for r in view_dice_rows],
    }
    view_summary_rows = []
    for name, values in view_groups.items():
        arr = np.asarray(values, dtype=np.float64)
        view_summary_rows.append({"view_group": name, "n": int(arr.size), "mean": float(arr.mean()),
                                  "std_population": float(arr.std(ddof=0)), "variance_population": float(arr.var(ddof=0)),
                                  "maximum": float(arr.max()), "minimum": float(arr.min())})
    write_csv(out / "view_dice_summary.csv", view_summary_rows)

    correlation_rows = []
    for metric in METRICS:
        x = np.asarray([row[f"{metric}_score"] for row in rows], dtype=np.float64)
        for target_name in TARGETS:
            y = np.asarray([row[target_name] for row in rows], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            rho = spearman(x, y)
            low, high, valid_bootstrap = bootstrap_spearman_ci(x, y, bootstrap_indices)
            correlation_rows.append({"metric": metric, "target": target_name, "spearman_rho": rho,
                                     "ci95_low": low, "ci95_high": high,
                                     "n_valid": int(valid.sum()), "n_invalid": int(len(rows)-valid.sum()),
                                     "bootstrap_replicates_valid": valid_bootstrap,
                                     "partial_spearman_control_M0": partial_spearman(x, y, m0_values) if metric != "M0" else float("nan")})
    write_csv(out / "metric_correlations.csv", correlation_rows)

    topk_rows: list[dict[str, Any]] = []
    selected_case_rows: list[dict[str, Any]] = []
    by_case = {row["case_id"]: row for row in rows}
    oracle_ids = sorted(rows, key=lambda r: (-r["joint_dice"], r["case_id"]))
    oracle_original_ids = sorted(rows, key=lambda r: (-r["original_dice"], r["case_id"]))
    for metric in METRICS:
        valid_rows = [r for r in rows if np.isfinite(float(r[f"{metric}_score"]))]
        ranked = sorted(valid_rows, key=lambda r: (-float(r[f"{metric}_score"]), r["case_id"]))
        for k in config["metrics"]["top_k"]:
            chosen = ranked[:min(int(k), len(ranked))]
            oracle = oracle_ids[:min(int(k), len(oracle_ids))]
            oracle_original = oracle_original_ids[:min(int(k), len(oracle_original_ids))]
            chosen_ids = {r["case_id"] for r in chosen}
            overlap = len(chosen_ids.intersection(r["case_id"] for r in oracle))
            dices = np.asarray([r["joint_dice"] for r in chosen], dtype=np.float64)
            original_dices = np.asarray([r["original_dice"] for r in chosen], dtype=np.float64)
            topk_rows.append({"metric": metric, "top_k_requested": int(k), "selected_n": len(chosen),
                              "valid_metric_cases": len(valid_rows), "mean_joint_dice": float(dices.mean()) if len(dices) else float("nan"),
                              "median_joint_dice": float(np.median(dices)) if len(dices) else float("nan"),
                              "fraction_dice_gt_0_7": float(np.mean(dices > 0.7)) if len(dices) else float("nan"),
                              "fraction_dice_lt_0_1": float(np.mean(dices < 0.1)) if len(dices) else float("nan"),
                              "mean_original_dice": float(original_dices.mean()) if len(original_dices) else float("nan"),
                              "median_original_dice": float(np.median(original_dices)) if len(original_dices) else float("nan"),
                              "original_dice_gt_0_7_fraction": float(np.mean(original_dices > 0.7)) if len(original_dices) else float("nan"),
                              "original_dice_lt_0_1_fraction": float(np.mean(original_dices < 0.1)) if len(original_dices) else float("nan"),
                              "empty_joint_predictions": int(sum(r["empty_joint_prediction"] for r in chosen)),
                              "empty_original_predictions": int(sum(r["empty_original_prediction"] for r in chosen)),
                              "mean_gt_foreground_voxels": float(np.mean([r["gt_foreground_voxels"] for r in chosen])) if chosen else float("nan"),
                              "oracle_joint_dice_topk_overlap": overlap,
                              "oracle_joint_overlap_fraction": overlap/max(1, len(chosen)),
                              "oracle_original_dice_topk_overlap": len(chosen_ids.intersection(r["case_id"] for r in oracle_original)),
                              "oracle_original_overlap_fraction": len(chosen_ids.intersection(r["case_id"] for r in oracle_original))/max(1, len(chosen))})
            for rank, row in enumerate(chosen, 1):
                selected_case_rows.append({"metric": metric, "top_k_requested": int(k), "rank": rank,
                                           "case_id": row["case_id"], "metric_score": row[f"{metric}_score"],
                                           "joint_dice": row["joint_dice"], "original_dice": row["original_dice"],
                                           "gt_foreground_voxels": row["gt_foreground_voxels"],
                                           "empty_joint_prediction": row["empty_joint_prediction"],
                                           "empty_original_prediction": row["empty_original_prediction"],
                                           "oracle_joint_dice_topk_member": int(row["case_id"] in {r["case_id"] for r in oracle}),
                                           "oracle_original_dice_topk_member": int(row["case_id"] in {r["case_id"] for r in oracle_original})})
    write_csv(out / "topk_screening_summary.csv", topk_rows)
    write_csv(out / "topk_selected_cases.csv", selected_case_rows)

    stratum_rows: list[dict[str, Any]] = []
    for stratum in sorted({r["m0_stratum"] for r in rows}):
        group = [r for r in rows if r["m0_stratum"] == stratum]
        for metric in METRICS:
            scores = np.asarray([r[f"{metric}_score"] for r in group], dtype=np.float64)
            dices = np.asarray([r["joint_dice"] for r in group], dtype=np.float64)
            stratum_rows.append({"m0_stratum": stratum, "metric": metric, "n": len(group),
                                 "m0_mean": float(np.mean([r["M0_score"] for r in group])),
                                 "metric_score_mean": float(np.mean(scores)), "joint_dice_mean": float(np.mean(dices)),
                                 "joint_dice_median": float(np.median(dices)),
                                 "spearman_with_joint_dice_within_stratum": spearman(scores, dices)})
    write_csv(out / "m0_strata_diagnostic.csv", stratum_rows)

    size_rows = []
    for metric in METRICS:
        score = np.asarray([r[f"{metric}_score"] for r in rows], dtype=np.float64)
        size = np.log10(1.0 + np.asarray([r["gt_foreground_voxels"] for r in rows], dtype=np.float64))
        size_rows.append({"metric": metric, "spearman_score_vs_log10_gt_volume": spearman(score, size),
                          "mean_gt_volume_all": float(np.mean([r["gt_foreground_voxels"] for r in rows]))})
    write_csv(out / "lesion_size_association.csv", size_rows)

    # Keep a compact audit of cases where a score is confidently wrong or
    # conservatively wrong. These are descriptions only, never tuning inputs.
    failure_rows = []
    for metric in METRICS:
        valid_rows = [r for r in rows if np.isfinite(float(r[f"{metric}_score"]))]
        descending = sorted(valid_rows, key=lambda r: (-float(r[f"{metric}_score"]), r["case_id"]))
        ascending = sorted(valid_rows, key=lambda r: (float(r[f"{metric}_score"]), r["case_id"]))
        high_score_bad = sorted((r for r in descending[:20] if r["joint_dice"] < 0.1), key=lambda r: (r["joint_dice"], -float(r[f"{metric}_score"])))[:5]
        low_score_good = sorted((r for r in ascending[:20] if r["joint_dice"] > 0.7), key=lambda r: (-r["joint_dice"], float(r[f"{metric}_score"])))[:5]
        for kind, examples in (("high_score_low_dice", high_score_bad), ("low_score_high_dice", low_score_good)):
            for row in examples:
                failure_rows.append({"metric": metric, "failure_type": kind, "case_id": row["case_id"],
                                     "metric_score": row[f"{metric}_score"], "original_dice": row["original_dice"],
                                     "joint_dice": row["joint_dice"], "gt_foreground_voxels": row["gt_foreground_voxels"],
                                     "original_foreground_voxels": row["original_foreground_voxels"],
                                     "joint_foreground_voxels": row["joint_foreground_voxels"],
                                     "empty_original_prediction": row["empty_original_prediction"],
                                     "empty_joint_prediction": row["empty_joint_prediction"]})
    write_csv(out / "failure_case_examples.csv", failure_rows or [{"note": "No examples met strict diagnostic criteria."}])

    # Scatter panels, coloring by GT foreground volume for visual bias diagnostics.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt_volume = np.asarray([r["gt_foreground_voxels"] for r in rows], dtype=np.float64)
    color = np.log10(1.0 + gt_volume)
    plot_map = {"original_dice": "Original Dice", "joint_dice": "Joint fused Dice",
                "all_view_dice_mean": "All-view mean Dice", "all_view_dice_variance": "All-view Dice variance"}
    for target in TARGETS:
        fig, axes = plt.subplots(4, 2, figsize=(11, 15), constrained_layout=True)
        for index, metric in enumerate(METRICS):
            axis = axes.flat[index]
            x = np.asarray([r[f"{metric}_score"] for r in rows], dtype=np.float64)
            y = np.asarray([r[target] for r in rows], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            sc = axis.scatter(x[valid], y[valid], c=color[valid], cmap="viridis", s=24, alpha=0.8)
            rho = spearman(x, y)
            axis.set_title(f"{metric}: Spearman={rho:.3f} (n={valid.sum()})")
            axis.set_xlabel(f"{metric} reliability score")
            axis.set_ylabel(plot_map[target])
            axis.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        fig.colorbar(sc, ax=list(axes.flat[:-1]), label="log10(1 + GT foreground voxels)", shrink=0.7)
        fig.savefig(out / f"scatter_{target}.png", dpi=170)
        plt.close(fig)

    source_valid = [r for r in csv_summary_rows if r["status"] == "valid"]
    source_diff = np.asarray([r["max_abs_component_difference"] for r in source_valid], dtype=np.float64)
    source_agreement = {"cases": len(source_valid), "missing_or_invalid_cases": len(rows)-len(source_valid),
                        "max_abs_uncertainty_component_difference": float(source_diff.max()) if len(source_diff) else None,
                        "mean_abs_uncertainty_component_difference": float(source_diff.mean()) if len(source_diff) else None,
                        "within_1e-4_cases": int(np.count_nonzero(source_diff <= 1e-4)),
                        "within_1e-3_cases": int(np.count_nonzero(source_diff <= 1e-3))}
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True).stdout.strip()
    manifest = {
        "protocol": "offline source-model inference and post-hoc GT diagnostic; no adaptation/training",
        "repo_revision_at_run": git_revision, "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model_checkpoint": file_identity(Path(config["model"]["checkpoint"])),
        "voxtell_root": str(voxtell_root), "voxtell_predictor_path": str((voxtell_root / "voxtell/inference/predictor.py").resolve()),
        "split_key": config["data"]["split_key"], "case_count": len(rows), "case_ids": cases,
        "model_updates": False, "gt_usage": "post-hoc Dice/selection diagnostics only; never used for M0-M6 computation, prediction, or filtering thresholds",
        "all_metrics_from_same_predictions": True, "same_frozen_model_and_prompts": True,
        "threshold": threshold, "inference": config["inference"], "augmentation": config["augmentation"],
        "prompts": config["model"]["prompts"], "fusion": "per-view Gaussian-logit sliding-window; M3 branch fusion uses repository _consensus temperature=0.15; final joint = 0.5 image + 0.5 prompt",
        "cache": {"directory": str(cache_dir), "independent_view_predictions_saved": True,
                  "cases": cache_manifest_cases, "note": "crop-space maps are indexed by saved bbox and are already returned to original evaluation orientation; this protocol has identity geometric transform"},
        "old_consensus_cache_audit": {"path": "outputs/ablation_consensus_cache", "num_cases": 101,
                                      "independent_view_predictions_saved": False,
                                      "reused": False,
                                      "reason": "metadata does not certify required step=0.5 Gaussian fusion and stores only fused probabilities/confidence"},
        "source_uncertainty_csv": str(Path(config["source_uncertainty_csv"]).resolve()),
        "source_uncertainty_reproduction": source_agreement,
        "metric_missing_or_invalid_cases": {metric: sum(not np.isfinite(float(r[f"{metric}_score"])) for r in rows) for metric in METRICS},
        "bootstrap": {"replicates": bootstrap_replicates, "seed": int(config["metrics"]["bootstrap_seed"]), "unit": "case resampling"},
        "source_images_sha256": None,
    }
    (out / "experiment_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    empty_counts = {"original_empty_cases": int(sum(r["empty_original_prediction"] for r in rows)),
                    "joint_empty_cases": int(sum(r["empty_joint_prediction"] for r in rows)),
                    "M1_zeroed_by_empty_gate": int(sum(r["empty_original_prediction"] for r in rows)),
                    "M1_changed_from_M0_cases": int(sum(abs(r["M1_score"]-r["M0_score"]) > 1e-12 for r in rows))}
    ranking_by_joint = sorted((r for r in correlation_rows if r["target"] == "joint_dice"),
                              key=lambda r: (-(r["spearman_rho"] if np.isfinite(r["spearman_rho"]) else -2), r["metric"]))
    topk_by_metric = {metric: next((r for r in topk_rows if r["metric"] == metric and r["top_k_requested"] == 20), None) for metric in METRICS}
    size_by_metric = {r["metric"]: r for r in size_rows}
    strata_by_metric = {}
    for metric in METRICS:
        values = [r["spearman_with_joint_dice_within_stratum"] for r in stratum_rows if r["metric"] == metric and np.isfinite(r["spearman_with_joint_dice_within_stratum"])]
        strata_by_metric[metric] = float(np.mean(values)) if values else float("nan")
    report_lines = [
        "# 无标签伪标签质量指标离线诊断",
        "",
        f"病例数：{len(rows)}（WCH_MRI145 DWI 训练 split）；模型和预测冻结，未运行 Adapt 训练。",
        "所有预测在 nnUNet 重定向后的完整评估空间中生成：192³ 滑窗、step=0.5、Gaussian 加权 logit 融合，再 sigmoid；图像增强只有强度扰动，几何逆变换为恒等。GT 只在预测及 M0–M6 计算完成后用于 Dice、病例排序诊断和病灶体积偏差分析。",
        "",
        "## 预测一致性与有效性",
        "",
        f"- 原 Consensus Cache 有 101 例融合概率和置信度，但无独立 view 预测，也没有足够元数据证明 step=0.5 Gaussian 协议；因此本实验没有将它与其他推理结果混用。逐 view 预测已重新生成并缓存。",
        f"- 现有不确定性 CSV 与新生成预测的组件差异审计：{source_agreement['cases']} 例有有效源值，最大绝对差 `{source_agreement['max_abs_uncertainty_component_difference']}`，均值 `{source_agreement['mean_abs_uncertainty_component_difference']}`；差异 ≤1e-4：{source_agreement['within_1e-4_cases']} 例，≤1e-3：{source_agreement['within_1e-3_cases']} 例。M0 使用重算的同一预测组分数；CSV 仅作可复现性对照。",
        f"- 缺失/无效指标病例数：{json.dumps({k:int(v) for k,v in manifest['metric_missing_or_invalid_cases'].items()}, ensure_ascii=False)}。M2/M3/M4 双方预测均空时得分为 0。",
        f"- 空预测：原图 {empty_counts['original_empty_cases']} 例，最终联合融合 {empty_counts['joint_empty_cases']} 例；M1 将原图空预测的 {empty_counts['M1_zeroed_by_empty_gate']} 例分数置 0，共改变 {empty_counts['M1_changed_from_M0_cases']} 例分数。",
        "",
        "## 视图真实 Dice 分布",
        "",
        "| 视图组 | n | 均值 | 标准差 | 方差 | 最大 | 最小 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in view_summary_rows:
        report_lines.append(f"| {r['view_group']} | {r['n']} | {r['mean']:.4f} | {r['std_population']:.4f} | {r['variance_population']:.4f} | {r['maximum']:.4f} | {r['minimum']:.4f} |")
    report_lines += ["", "## 与联合融合 Dice 的 Spearman 相关性", "", "| 指标 | rho [病例 bootstrap 95% CI] | n | Top-20 平均联合 Dice | Oracle Top-20 重合 | M0 条件偏相关 |", "|---|---:|---:|---:|---:|---:|"]
    for r in ranking_by_joint:
        top = topk_by_metric[r["metric"]]
        report_lines.append(f"| {r['metric']} | {r['spearman_rho']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] | {r['n_valid']} | {top['mean_joint_dice']:.3f} | {top['oracle_joint_overlap_fraction']:.1%} | {r['partial_spearman_control_M0']:.3f} |")
    report_lines += ["", "## 相近 M0 稳定性下的区分能力", "", "病例按 M0 分为 5 个等频组；`m0_strata_diagnostic.csv` 给出每组均值/中位数和每项分数与联合 Dice 的组内 Spearman。`metric_correlations.csv` 另给对 M0 排名控制后的偏 Spearman。组内相关性若接近 0，说明指标没有稳定地提供 M0 之外的区分信息。", "", "| 指标 | 五组内 Spearman 均值 | M0 控制偏 Spearman |", "|---|---:|---:|"]
    corr_by_metric = {r["metric"]: r for r in correlation_rows if r["target"] == "joint_dice"}
    for metric in METRICS:
        report_lines.append(f"| {metric} | {strata_by_metric[metric]:.3f} | {corr_by_metric[metric]['partial_spearman_control_M0']:.3f} |")
    report_lines += ["", "## Top-K 筛选和病灶大小偏差", "", "完整结果见 `topk_screening_summary.csv`，含原图及联合 Dice、空预测数和两种 Dice Oracle Top-K 重合率。病灶大小关联以 `lesion_size_association.csv` 中的 score 与 log10(1+GT 前景体素) Spearman 表示；入选病例体积另见 Top-K 汇总。", "", "| 指标 | score 与 GT 体积 Spearman | Top-20 平均 GT 前景体素 | Top-20 平均联合 Dice |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        top = topk_by_metric[metric]
        report_lines.append(f"| {metric} | {size_by_metric[metric]['spearman_score_vs_log10_gt_volume']:.3f} | {top['mean_gt_foreground_voxels']:.1f} | {top['mean_joint_dice']:.3f} |")
    report_lines += ["", "## 典型失败病例", "", "`failure_case_examples.csv` 列出每项指标中 Top-20 高分但联合 Dice<0.1，以及 Bottom-20 低分但联合 Dice>0.7 的例子；该筛选只用于描述，不反馈到指标定义或阈值。", "", "## 图表与数据文件", "", "- `per_case_metrics.csv`：逐病例 M0–M6、原图/分支/联合 Dice、逐视图 Dice 与前景数。",
                    "- `per_view_dice.csv`、`view_dice_summary.csv`：逐 view 质量和 Image/Prompt/全部 view 统计。",
                    "- `metric_correlations.csv`：四种目标相关性及 bootstrap 95% CI。",
                    "- `topk_screening_summary.csv`、`topk_selected_cases.csv`：Top-20/40/59 筛选。",
                    "- `m0_strata_diagnostic.csv`、`lesion_size_association.csv`、`failure_case_examples.csv`：增量信息、病灶大小偏差和失败病例。",
                    "- `view_prediction_cache/`：逐病例独立概率图与空间/推理元信息，供离线复算。",
                    "- `scatter_*.png`：M0–M6 与四种目标的相关散点图，点颜色表示 GT 病灶大小。",
                    "",
                    "GT Dice 与病灶体积仅为离线诊断标签；任何指标公式、阈值、Top-K 大小均未根据 GT 调整。"]
    (out / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Completed offline diagnostic: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
