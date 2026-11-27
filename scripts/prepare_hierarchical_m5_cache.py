from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from cue_sfda.data import CaseRecord
from cue_sfda.method.dynamic_hard import (
    DynamicHardCache,
    consensus,
    source_checkpoint_signature,
)


def case_id(name: str) -> str:
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def entropy(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability.astype(np.float32, copy=False), 1e-6, 1.0 - 1e-6)
    return -(value * np.log2(value) + (1.0 - value) * np.log2(1.0 - value))


def padded(roi_array: np.ndarray, padded_shape: tuple[int, ...], roi_slicer: tuple[slice, ...]) -> np.ndarray:
    if roi_array.ndim == 3:
        roi_array = roi_array[None]
    result = np.zeros((roi_array.shape[0], *padded_shape), dtype=roi_array.dtype)
    result[(slice(None), *roi_slicer)] = roi_array
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build M5's initial Dynamic-hard cache from existing benchmark probabilities"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/hierarchical_m5.yaml"))
    parser.add_argument("--probability-dir", type=Path, default=Path("analysis/hierarchical_uncertainty/stage1_image_level/probabilities"))
    parser.add_argument("--pixel-weight-dir", type=Path, default=Path("analysis/hierarchical_dynamic/fixed_pixel_uncertainty"))
    parser.add_argument("--confidence-csv", type=Path, default=Path("analysis/hierarchical_dynamic/image_confidence.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("/data/hilab/datasets/WCH_MRI145/DWI/images"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dynamic = config["dynamic_hard"]
    confidence_rows = {}
    with args.confidence_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            confidence_rows[str(row["case_id"])] = row
    if len(confidence_rows) != 101:
        raise ValueError(f"Expected 101 initial confidence rows, found {len(confidence_rows)}")

    cache = DynamicHardCache(dynamic["cache_dir"])
    case_ids = sorted(confidence_rows)
    case_confidences = {}
    for cid in case_ids:
        case_name = f"{cid}.nii.gz"
        probability_path = args.probability_dir / f"{cid}.npz"
        pixel_path = args.pixel_weight_dir / f"{cid}.npz"
        image_path = args.image_dir / case_name
        with np.load(probability_path, allow_pickle=False) as source:
            image_views = source["image_views"].astype(np.float32, copy=False)
            prompt_views = source["prompt_views"].astype(np.float32, copy=False)
            original = source["original"].astype(np.float32, copy=False)[0]
        with np.load(pixel_path, allow_pickle=False) as pixel:
            metadata = json.loads(str(pixel["metadata"].item()))
            pixel_weight = pixel["weight"].astype(np.float32, copy=False)
        padded_shape = tuple(int(value) for value in metadata["padded_shape"])
        roi_slicer = tuple(slice(*axis) for axis in metadata["roi_slicer"])
        image_probability, image_voxel_confidence, _ = consensus(
            image_views[:, None], float(dynamic["image_temperature"])
        )
        prompt_probability, prompt_voxel_confidence, _ = consensus(
            prompt_views[:, None], float(dynamic["prompt_temperature"])
        )
        joint_probability = (0.5 * image_probability + 0.5 * prompt_probability).astype(np.float32)
        joint_confidence = np.sqrt(
            np.clip(image_voxel_confidence * prompt_voxel_confidence, 0.0, 1.0)
        ).astype(np.float32)
        if pixel_weight.shape[-3:] == original.shape:
            pixel_weight_roi = pixel_weight
        elif pixel_weight.shape[-3:] == padded_shape:
            pixel_weight_roi = pixel_weight[(slice(None), *roi_slicer)]
        else:
            raise ValueError(
                f"Pixel map geometry mismatch for {cid}: "
                f"weight={pixel_weight.shape[-3:]}, "
                f"ROI={original.shape}, padded={padded_shape}"
            )
        if pixel_weight_roi.shape[-3:] != original.shape:
            raise ValueError(
                f"Pixel map ROI mismatch after geometry crop for {cid}: "
                f"{pixel_weight_roi.shape[-3:]} vs {original.shape}"
            )
        confidence = confidence_rows[cid]
        c_image = float(confidence["C_image"])
        c_prompt = float(confidence["C_prompt"])
        case_confidence = float(confidence["C_case"])
        original_empty = not bool(np.any(original >= float(config["method"]["probability_threshold"])))
        stat = image_path.stat()
        padded_probability = padded(joint_probability, padded_shape, roi_slicer)
        padded_confidence = padded(joint_confidence, padded_shape, roi_slicer)
        padded_weight = padded(pixel_weight_roi, padded_shape, roi_slicer)
        quality_mask = padded(
            original >= float(config["method"]["probability_threshold"]),
            padded_shape,
            roi_slicer,
        ).astype(np.uint8)
        geometry = {
            "case_id": case_name,
            "image_size": int(stat.st_size),
            "image_mtime_ns": int(stat.st_mtime_ns),
            "padded_shape": list(padded_shape),
            "roi_shape": [int(value) for value in original.shape],
            "roi_slicer": [[int(s.start or 0), int(s.stop or 0), int(s.step or 1)] for s in roi_slicer],
            "cache_version": 0,
            "teacher_type": "frozen_source",
            "teacher_iteration": None,
            "quality_protocol": "benchmark_gaussian",
            "original_prediction_empty": original_empty,
            "image_case_confidence": c_image,
            "prompt_case_confidence": c_prompt,
            "case_confidence_raw": 0.4 * c_image + 0.6 * c_prompt,
            "case_confidence": case_confidence,
            "predicted_voxels": int(np.count_nonzero(original >= float(config["method"]["probability_threshold"]))),
        }
        cache.save_case(
            CaseRecord(name=case_name, image_path=image_path),
            0,
            padded_probability,
            padded_confidence,
            geometry,
            pixel_weight=padded_weight,
            quality_prediction_mask=quality_mask,
        )
        case_confidences[case_name] = case_confidence

    metadata = {
        "schema_version": 1,
        "cache_version": 0,
        "teacher_type": "frozen_source",
        "num_cases": len(case_ids),
        "case_ids": [f"{cid}.nii.gz" for cid in case_ids],
        "num_views": int(config["method"]["num_views"]),
        "prompts": [str(prompt) for prompt in dynamic["prompts"]],
        "standard_prompt": str(dynamic["standard_prompt"]),
        "image_temperature": float(dynamic["image_temperature"]),
        "prompt_temperature": float(dynamic["prompt_temperature"]),
        "refresh_tile_step_size": float(dynamic["refresh_tile_step_size"]),
        "teacher_iteration": None,
        "source_checkpoint": source_checkpoint_signature(config["model"]),
        "quality_protocol": "benchmark_gaussian",
        "source_probabilities": str(args.probability_dir.resolve()),
    }
    cache.write_version_metadata(0, metadata)
    confidence_path = cache.version_dir(0) / "case_confidences.json"
    confidence_path.write_text(
        json.dumps(case_confidences, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache.activate_version(0)
    print(f"built M5 version-0 cache for {len(case_ids)} cases from cached benchmark probabilities")


if __name__ == "__main__":
    main()
