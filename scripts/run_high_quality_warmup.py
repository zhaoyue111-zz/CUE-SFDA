#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cue_sfda.config import load_config
from cue_sfda.data import load_split
from cue_sfda.engine.high_quality_warmup import (
    load_and_validate_routing,
    run_high_quality_warmup,
)
from cue_sfda.modeling import build_runtime
from cue_sfda.utils import configure_logging, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="High-only VoxTell warm-up")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-id", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = str(args.output_dir)
    if args.gpu_id is not None:
        config["experiment"]["gpu_id"] = int(args.gpu_id)
    if int(config["warmup"].get("epochs", 0)) != 20:
        raise ValueError("High-quality warm-up is defined as exactly 20 epochs")
    if list(config["warmup"].get("evaluation_epochs", [])) != [0, 5, 10, 15, 20]:
        raise ValueError("Evaluation epochs must be [0, 5, 10, 15, 20]")

    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    logger = configure_logging(output_dir)
    seed_everything(int(config["experiment"]["seed"]))
    cases, _ = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    if len(cases) != 101:
        raise ValueError(f"Expected exactly 101 train cases, got {len(cases)}")
    groups, _ = load_and_validate_routing(config["warmup"]["routing_csv"], cases)
    high = sorted(case.name for case in cases if groups[case.name] == "high")
    low = sorted(case.name for case in cases if groups[case.name] == "low")
    logger.info(
        "Validated saved route before loading VoxTell: source=%s High=%d Low=%d Skip=0\n"
        "High IDs: %s\nLow IDs: %s",
        Path(config["warmup"]["routing_csv"]).resolve(),
        len(high),
        len(low),
        ",".join(high),
        ",".join(low),
    )

    # Reuse the exact prompt embedding, but place a separate copy in this run's
    # cache so the experiment never shares mutable output artifacts.
    source_cache = Path(config["warmup"].get("prompt_cache_source", ""))
    prompt_cache = output_dir / "prompt_cache"
    cached_prompt = sorted(source_cache.glob("liver_cancer_*.pt")) if source_cache.is_dir() else []
    if cached_prompt:
        prompt_cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_prompt[0], prompt_cache / cached_prompt[0].name)

    runtime = build_runtime(config, prompt_cache)
    trainable = sum(parameter.numel() for parameter in runtime.trainable_parameters)
    total = sum(parameter.numel() for parameter in runtime.model.parameters())
    logger.info(
        "VoxTell ready from original pretrained weights: LoRA modules=%d trainable=%d "
        "total=%d device=%s; no adapted checkpoint loaded",
        runtime.lora_modules,
        trainable,
        total,
        runtime.device,
    )
    run_high_quality_warmup(
        config,
        runtime,
        cases,
        args.gt_path,
        output_dir,
        logger,
    )


if __name__ == "__main__":
    main()
