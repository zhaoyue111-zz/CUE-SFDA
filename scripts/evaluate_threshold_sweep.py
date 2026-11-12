#!/usr/bin/env python3
"""DWI-only threshold diagnostic; labels are used for evaluation only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    ZScoreNormalization,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cue_sfda.config import load_config  # noqa: E402
from cue_sfda.data import load_split  # noqa: E402
from cue_sfda.evaluation import binary_metrics, predict_volume  # noqa: E402
from cue_sfda.modeling import build_runtime  # noqa: E402
from cue_sfda.utils import seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    parser.add_argument("--empty-fallback-topk", type=int, nargs="+", default=[])
    parser.add_argument("--component-min-sizes", type=int, nargs="+", default=[])
    parser.add_argument("--largest-component", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["experiment"]["gpu_id"] = args.gpu_id
    seed_everything(int(config["experiment"]["seed"]))
    _, cases = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    cache_dir = args.output.parent / f"{args.output.stem}_prompt_cache"
    runtime = build_runtime(config, cache_dir)
    runtime.model.eval()
    reader = NibabelIOWithReorient()
    normalizer = ZScoreNormalization(intensityproperties={})
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        image, _ = reader.read_images([str(case.image_path)])
        label, _ = reader.read_seg(str(args.gt_path / case.name))
        ground_truth = label == int(config["evaluation"]["label_value"])
        probability = predict_volume(runtime, image, normalizer, config)
        for threshold in args.thresholds:
            prediction = probability >= threshold
            dice, iou = binary_metrics(prediction, ground_truth)
            rows.append(
                {
                    "case": case.name,
                    "mode": "threshold",
                    "threshold": threshold,
                    "fallback_topk": 0,
                    "dice": dice,
                    "iou": iou,
                    "predicted_voxels": int(prediction.sum()),
                    "ground_truth_voxels": int(ground_truth.sum()),
                }
            )
            components, component_count = ndimage.label(prediction)
            component_sizes = np.bincount(components.reshape(-1))
            for minimum in args.component_min_sizes:
                keep = component_sizes >= int(minimum)
                keep[0] = False
                filtered = keep[components]
                dice, iou = binary_metrics(filtered, ground_truth)
                rows.append(
                    {
                        "case": case.name,
                        "mode": f"components_ge_{minimum}",
                        "threshold": threshold,
                        "fallback_topk": 0,
                        "dice": dice,
                        "iou": iou,
                        "predicted_voxels": int(filtered.sum()),
                        "ground_truth_voxels": int(ground_truth.sum()),
                    }
                )
            if args.largest_component:
                if component_count:
                    largest = 1 + int(np.argmax(component_sizes[1:]))
                    filtered = components == largest
                else:
                    filtered = prediction
                dice, iou = binary_metrics(filtered, ground_truth)
                rows.append(
                    {
                        "case": case.name,
                        "mode": "largest_component",
                        "threshold": threshold,
                        "fallback_topk": 0,
                        "dice": dice,
                        "iou": iou,
                        "predicted_voxels": int(filtered.sum()),
                        "ground_truth_voxels": int(ground_truth.sum()),
                    }
                )
        primary = probability >= 0.5
        for topk in args.empty_fallback_topk:
            prediction = primary.copy()
            if not prediction.any():
                flat = probability.reshape(-1)
                count = min(int(topk), flat.size)
                selected = np.argpartition(flat, -count)[-count:]
                prediction.reshape(-1)[selected] = True
            dice, iou = binary_metrics(prediction, ground_truth)
            rows.append(
                {
                    "case": case.name,
                    "mode": "empty_topk",
                    "threshold": 0.5,
                    "fallback_topk": topk,
                    "dice": dice,
                    "iou": iou,
                    "predicted_voxels": int(prediction.sum()),
                    "ground_truth_voxels": int(ground_truth.sum()),
                }
            )
        print(f"{index}/{len(cases)} {case.name}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for threshold in args.thresholds:
        selected = [
            row
            for row in rows
            if row["mode"] == "threshold" and row["threshold"] == threshold
        ]
        mean_dice = float(np.mean([row["dice"] for row in selected]))
        empty = sum(int(row["predicted_voxels"]) == 0 for row in selected)
        print(f"threshold={threshold:.3f} mean_dice={mean_dice:.6f} empty={empty}")
    for topk in args.empty_fallback_topk:
        selected = [
            row
            for row in rows
            if row["mode"] == "empty_topk" and row["fallback_topk"] == topk
        ]
        mean_dice = float(np.mean([row["dice"] for row in selected]))
        print(f"empty_fallback_topk={topk} mean_dice={mean_dice:.6f}")
    postprocess_modes = [
        *(f"components_ge_{value}" for value in args.component_min_sizes),
        *(("largest_component",) if args.largest_component else ()),
    ]
    for threshold in args.thresholds:
        for mode in postprocess_modes:
            selected = [
                row
                for row in rows
                if row["mode"] == mode and row["threshold"] == threshold
            ]
            mean_dice = float(np.mean([row["dice"] for row in selected]))
            print(
                f"threshold={threshold:.3f} mode={mode} mean_dice={mean_dice:.6f}"
            )


if __name__ == "__main__":
    main()
