#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cue_sfda.config import apply_overrides, load_config
from cue_sfda.data import TargetVolumeStore, load_split
from cue_sfda.method.augmentations import StrongViewGenerator
from cue_sfda.method.consensus import ConsensusCache, ConsensusCacheBuilder
from cue_sfda.modeling.voxtell import build_runtime, load_prompt_embedding
from cue_sfda.utils import configure_logging, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-free VoxTell consensus cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cache_build"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)
    config.setdefault("consensus", {})["mode"] = "joint"
    output = args.output_dir.resolve()
    logger = configure_logging(output)
    seed_everything(int(config["experiment"]["seed"]))
    train_cases, _ = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    train_cases = [case for index, case in enumerate(train_cases) if index % args.num_shards == args.shard_index]
    runtime = build_runtime(config, output / "prompt_cache")
    prompts = [str(v) for v in config["consensus"]["prompts"]]
    standard = str(config["consensus"].get("standard_prompt", config["model"]["prompt"]))
    embeddings = {standard: runtime.embedding}
    for prompt in prompts:
        embeddings.setdefault(prompt, load_prompt_embedding(config, prompt, output / "prompt_cache", runtime.device))
    store = TargetVolumeStore(tuple(int(v) for v in config["data"]["patch_size"]))
    views = StrongViewGenerator(config["method"], int(config["experiment"]["seed"]) + 10000)
    builder = ConsensusCacheBuilder(config, runtime, store, views, embeddings, logger)
    builder.disable_lora = True
    builder.build(train_cases, ConsensusCache(args.cache_dir), cycle=0, metadata_extra={"routing_frozen": True, "shard_index": args.shard_index, "num_shards": args.num_shards})


if __name__ == "__main__":
    main()
