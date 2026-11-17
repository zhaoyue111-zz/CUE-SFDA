from __future__ import annotations

import argparse
from pathlib import Path

from cue_sfda.config import load_config
from cue_sfda.data import load_split
from cue_sfda.method.high_pseudo_cache import build_initial_high_pseudo_cache
from cue_sfda.modeling import build_runtime
from cue_sfda.utils import configure_logging, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the shared frozen-source High pseudo-label cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    seed_everything(int(config["experiment"]["seed"]))
    train_cases, _ = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    logger.info("Building frozen source cache for %d High cases", len(train_cases))
    runtime = build_runtime(config, output_dir / "prompt_cache")
    logger.info("Frozen VoxTell ready: prompt=%s device=%s", config["model"]["prompt"], runtime.device)
    build_initial_high_pseudo_cache(config, runtime, train_cases, args.cache_dir, logger)
    logger.info("Cache complete: %s", args.cache_dir.resolve())


if __name__ == "__main__":
    main()
