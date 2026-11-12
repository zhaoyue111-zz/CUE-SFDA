#!/usr/bin/env python3
"""Evaluate one VoxTell/LoRA checkpoint with both fixed DWI prompts."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

import torch

# python scripts/evaluate_both_prompts.py \
#     --checkpoint outputs/dwi_only_liver_cancer/checkpoints/best_iteration_000700_dice_0.330541.pt \
#     --output-dir outputs/dwi_only_liver_cancer/evaluation_both_with_liver_cancer_ckpt \
#     --gpu-id 0

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cue_sfda.config import load_config  # noqa: E402
from cue_sfda.data import load_split  # noqa: E402
from cue_sfda.engine.checkpoint import load_checkpoint  # noqa: E402
from cue_sfda.evaluation import evaluate  # noqa: E402
from cue_sfda.modeling import (  # noqa: E402
    build_runtime,
    load_prompt_embedding,
)
from cue_sfda.utils import configure_logging, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one checkpoint once and evaluate the fixed prompts 'liver' "
            "and 'liver cancer'."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "dwi_liver.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--liver-gt-path",
        type=Path,
        default=Path("/data/hilab/datasets/WCH_MRI145/DWI/liver"),
    )
    parser.add_argument(
        "--cancer-gt-path",
        type=Path,
        default=Path("/data/hilab/datasets/WCH_MRI145/DWI/labels"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint.parent.parent / "evaluation_both_with_liver_ckpt"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)

    base_config = load_config(args.config)
    base_config["experiment"]["gpu_id"] = int(args.gpu_id)
    base_config["model"]["prompt"] = "liver"
    seed_everything(int(base_config["experiment"]["seed"]))
    _, test_cases = load_split(
        base_config["data"]["image_dir"],
        base_config["data"]["split_json"],
    )

    cache_dir = output_dir / "prompt_cache"
    device = torch.device(f"cuda:{int(args.gpu_id)}")
    # Encode/cache the second prompt before the 3-D model occupies GPU memory.
    cancer_embedding = load_prompt_embedding(
        base_config,
        "liver cancer",
        cache_dir,
        device,
    )
    runtime = build_runtime(base_config, cache_dir)
    load_checkpoint(checkpoint, runtime.model)
    runtime.model.eval()
    logger.info("Loaded checkpoint once: %s", checkpoint)

    liver_config = deepcopy(base_config)
    liver_config["model"]["prompt"] = "liver"
    liver_summary = evaluate(
        liver_config,
        runtime,
        test_cases,
        args.liver_gt_path,
        output_dir / "liver",
        logger,
        save_predictions=True,
    )

    runtime.embedding = cancer_embedding
    cancer_config = deepcopy(base_config)
    cancer_config["model"]["prompt"] = "liver cancer"
    cancer_summary = evaluate(
        cancer_config,
        runtime,
        test_cases,
        args.cancer_gt_path,
        output_dir / "liver_cancer",
        logger,
        save_predictions=True,
    )

    rows = [
        {
            "task": "liver",
            "prompt": "liver",
            **liver_summary,
            "prediction_dir": str(output_dir / "liver" / "predictions"),
        },
        {
            "task": "liver_cancer",
            "prompt": "liver cancer",
            **cancer_summary,
            "prediction_dir": str(output_dir / "liver_cancer" / "predictions"),
        },
    ]
    with (output_dir / "metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "checkpoint": str(checkpoint),
        "config": str(Path(args.config).expanduser().resolve()),
        "gpu_id": int(args.gpu_id),
        "tasks": rows,
    }
    with (output_dir / "evaluation_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    logger.info("Combined metrics: %s", output_dir / "metrics_summary.csv")


if __name__ == "__main__":
    main()
