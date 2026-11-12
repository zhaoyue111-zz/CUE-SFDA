#!/usr/bin/env python3
"""Evaluation-only decoder-stage diagnostic for the frozen VoxTell model.

Ground-truth masks are read only after inference and are never exposed to the
model or used to update parameters.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    ZScoreNormalization,
)

from cue_sfda.config import load_config
from cue_sfda.data import load_split
from cue_sfda.evaluation import binary_metrics
from cue_sfda.modeling import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--anatomy-predictions", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def predict_stages(runtime, image: np.ndarray, config: dict) -> list[np.ndarray]:
    normalizer = ZScoreNormalization(intensityproperties={})
    data = image.astype(np.float32, copy=True)
    original_shape = data.shape[1:]
    data, _, bounding_box = crop_to_nonzero(data, None)
    data = normalizer.run(data, None)
    data_tensor = torch.from_numpy(np.ascontiguousarray(data)).float()
    patch_size = tuple(int(value) for value in config["data"]["patch_size"])
    padded, revert_slicer = pad_nd_image(
        data_tensor, patch_size, "constant", {"value": 0}, True, None
    )
    shape = tuple(int(value) for value in padded.shape[1:])
    steps = compute_steps_for_sliding_window(
        shape, patch_size, float(config["evaluation"]["tile_step_size"])
    )
    dtype = torch.float16
    accumulators = [torch.zeros(shape, dtype=dtype, device=runtime.device) for _ in range(5)]
    normalization = torch.zeros(shape, dtype=dtype, device=runtime.device)
    gaussian = compute_gaussian(
        patch_size, sigma_scale=1 / 8, value_scaling_factor=10,
        dtype=dtype, device=runtime.device
    )
    for starts in ((d, h, w) for d in steps[0] for h in steps[1] for w in steps[2]):
        spatial_slice = tuple(
            slice(start, start + size) for start, size in zip(starts, patch_size, strict=True)
        )
        tile = padded[(slice(None), *spatial_slice)].unsqueeze(0).to(runtime.device)
        context = torch.autocast("cuda", dtype=torch.float16) if runtime.device.type == "cuda" else nullcontext()
        with context:
            outputs = runtime.model(tile, runtime.embedding, return_decoder_outputs=True)
        if not isinstance(outputs, list) or len(outputs) != 5:
            raise RuntimeError("VoxTell did not return five decoder stages")
        for index, logits in enumerate(outputs):
            aligned = F.interpolate(
                logits.float(), size=patch_size, mode="trilinear", align_corners=False
            )[0, 0].to(dtype)
            accumulators[index][spatial_slice] += aligned * gaussian
        normalization[spatial_slice] += gaussian
    probabilities = []
    for accumulator in accumulators:
        probability = torch.sigmoid((accumulator / normalization.clamp_min(1e-6)).float())
        cropped = probability[revert_slicer[1:]].cpu().numpy()
        restored = np.zeros(original_shape, dtype=np.float32)
        probabilities.append(insert_crop_into_image(restored, cropped, bounding_box))
    return probabilities


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["experiment"]["gpu_id"] = args.gpu_id
    _, cases = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    runtime = build_runtime(config, args.output.parent / "prompt_cache")
    reader = NibabelIOWithReorient()
    thresholds = [round(value, 2) for value in np.arange(0.3, 0.71, 0.05)]
    scores = {f"d{5-index}_t{threshold:.2f}": [] for index in range(5) for threshold in thresholds}
    scores.update({f"d5_d{stage}_avg_t{threshold:.2f}": [] for stage in range(2, 5) for threshold in thresholds})
    quantiles = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998]
    if args.anatomy_predictions:
        scores.update({f"anatomy_q{quantile:.3f}": [] for quantile in quantiles})
        scores.update({f"anatomy_union_q{quantile:.3f}": [] for quantile in quantiles})
    for case_index, case in enumerate(cases, 1):
        image, _ = reader.read_images([str(case.image_path)])
        label, _ = reader.read_seg(str(args.gt_path / case.name))
        ground_truth = label == int(config["evaluation"]["label_value"])
        probabilities = predict_stages(runtime, image, config)
        final = probabilities[0]
        if args.anatomy_predictions:
            anatomy, _ = reader.read_seg(str(args.anatomy_predictions / case.name))
            anatomy = np.squeeze(anatomy > 0)
            for quantile in quantiles:
                if anatomy.any():
                    cutoff = float(np.quantile(final[anatomy], quantile))
                    candidate = (final >= cutoff) & anatomy
                else:
                    candidate = np.zeros_like(anatomy)
                dice, _ = binary_metrics(candidate, ground_truth)
                scores[f"anatomy_q{quantile:.3f}"].append(dice)
                dice, _ = binary_metrics(candidate | (final >= 0.5), ground_truth)
                scores[f"anatomy_union_q{quantile:.3f}"].append(dice)
        for index, probability in enumerate(probabilities):
            for threshold in thresholds:
                dice, _ = binary_metrics(probability >= threshold, ground_truth)
                scores[f"d{5-index}_t{threshold:.2f}"].append(dice)
        for index, stage in enumerate(range(4, 1, -1), start=1):
            probability = 0.5 * (final + probabilities[index])
            for threshold in thresholds:
                dice, _ = binary_metrics(probability >= threshold, ground_truth)
                scores[f"d5_d{stage}_avg_t{threshold:.2f}"].append(dice)
        print(f"{case_index}/{len(cases)} {case.name}", flush=True)
    summary = {name: float(np.mean(values)) for name, values in scores.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, value in sorted(summary.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
