#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cue_sfda.config import load_config, validate_config
from cue_sfda.data import load_split
from cue_sfda.engine.high_quality_warmup import load_and_validate_routing
from cue_sfda.engine.low_only_dynamic_hard import run_low_only_dynamic_hard_warmup
from cue_sfda.modeling import build_runtime
from cue_sfda.utils import configure_logging, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-only 20-epoch Warm-up using M1 Dynamic-hard supervision"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-id", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = str(args.output_dir)
        config["dynamic_hard"]["cache_dir"] = str(args.output_dir / "dynamic_hard_cache")
    if args.gpu_id is not None:
        config["experiment"]["gpu_id"] = int(args.gpu_id)
    schedule = config["low_only_warmup"]
    if int(schedule.get("epochs", 0)) != 20:
        raise ValueError("Low-only Dynamic-hard Warm-up requires exactly 20 epochs")
    if list(schedule.get("evaluation_epochs", [])) != [0, 5, 10, 15, 20]:
        raise ValueError("Evaluation epochs must be [0, 5, 10, 15, 20]")
    if str(config.get("dynamic_hard", {}).get("mode")) != "dynamic_hard":
        raise ValueError("This experiment requires dynamic_hard.mode=dynamic_hard")

    output_dir = Path(config["experiment"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    cache_dir = Path(config["dynamic_hard"]["cache_dir"]).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (ROOT / cache_dir).resolve()
    historical = (ROOT / "outputs/ablation/M1_dynamic_hard").resolve()
    historical_cache = historical / "dynamic_hard_cache"
    if output_dir == historical or historical in output_dir.parents:
        raise ValueError("Choose an independent output directory; preserve historical M1 results")
    if (
        cache_dir == historical_cache
        or historical_cache in cache_dir.parents
        or cache_dir == historical
        or historical in cache_dir.parents
    ):
        raise ValueError("Choose an independent Dynamic-hard cache directory")
    config["experiment"]["output_dir"] = str(output_dir)
    config["dynamic_hard"]["cache_dir"] = str(cache_dir)
    validate_config(config)

    cases, _ = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    if len(cases) != 101:
        raise ValueError(f"Expected 101 train cases, got {len(cases)}")
    groups, _ = load_and_validate_routing(schedule["routing_csv"], cases)
    high_ids = sorted(case.name for case in cases if groups[case.name] == "high")
    low_ids = sorted(case.name for case in cases if groups[case.name] == "low")
    if len(high_ids) != 59 or len(low_ids) != 42:
        raise ValueError(f"Expected saved High=59/Low=42 route, got {len(high_ids)}/{len(low_ids)}")

    logger = configure_logging(output_dir)
    seed_everything(int(config["experiment"]["seed"]))
    logger.info(
        "Validated M1 saved routing before loading VoxTell: High=59 Low=42; "
        "only Low IDs train: %s",
        ",".join(low_ids),
    )

    source_prompt_cache = historical / "prompt_cache"
    prompt_cache = output_dir / "prompt_cache"
    if source_prompt_cache.is_dir():
        prompt_cache.mkdir(parents=True, exist_ok=True)
        for cache_file in source_prompt_cache.glob("*.pt"):
            shutil.copy2(cache_file, prompt_cache / cache_file.name)
    runtime = build_runtime(config, prompt_cache)
    trainable = sum(parameter.numel() for parameter in runtime.trainable_parameters)
    logger.info(
        "Original pretrained VoxTell initialized; LoRA modules=%d trainable=%d; "
        "no adapted checkpoint loaded",
        runtime.lora_modules,
        trainable,
    )
    run_low_only_dynamic_hard_warmup(
        config,
        runtime,
        cases,
        args.gt_path,
        output_dir,
        logger,
    )


if __name__ == "__main__":
    main()
