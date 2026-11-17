from __future__ import annotations

import argparse
from pathlib import Path

from cue_sfda.config import load_config
from cue_sfda.data import load_split
from cue_sfda.engine.checkpoint import load_checkpoint
from cue_sfda.evaluation import evaluate
from cue_sfda.modeling import build_runtime
from cue_sfda.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one fixed research_v2 checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config["experiment"]["gpu_id"] = args.gpu_id
    output_dir = args.output_dir.resolve()
    logger = configure_logging(output_dir)
    _, test_cases = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    runtime = build_runtime(config, output_dir / "prompt_cache")
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint, runtime.model)
        logger.info("Loaded fixed checkpoint iteration=%s from %s", state.get("iteration"), args.checkpoint)
    else:
        logger.info("Evaluating frozen VoxTell source weights")
    evaluate(config, runtime, test_cases, args.gt_path, output_dir / "evaluation", logger, save_predictions=False)


if __name__ == "__main__":
    main()
