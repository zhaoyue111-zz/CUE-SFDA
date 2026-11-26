from __future__ import annotations

import argparse
import json
from pathlib import Path

from cue_sfda.config import apply_overrides, load_config
from cue_sfda.data import load_split
from cue_sfda.engine import CUETrainer
from cue_sfda.engine.checkpoint import load_checkpoint
from cue_sfda.evaluation import evaluate
from cue_sfda.modeling import build_runtime
from cue_sfda.utils import configure_logging, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUE-SFDA adaptation with VoxTell")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser(
        "train", help="Adapt VoxTell on unlabeled target data"
    )
    _add_common_arguments(train_parser)
    train_parser.add_argument("--gt-path", type=Path)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--coverage-completion",
        action="store_true",
        help=(
            "On an explicitly resumed Dynamic-hard run, sample unseen primary "
            "cases uniformly until coverage completes"
        ),
    )
    train_parser.add_argument("--confidence-partition", type=Path)
    train_parser.add_argument("--skip-eval", action="store_true")
    evaluation_parser = subparsers.add_parser(
        "evaluate", help="Evaluate an adapted LoRA checkpoint"
    )
    _add_common_arguments(evaluation_parser)
    evaluation_parser.add_argument(
        "--checkpoint", type=Path,
        help="LoRA checkpoint; omit to evaluate the frozen VoxTell baseline",
    )
    evaluation_parser.add_argument("--gt-path", type=Path, required=True)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--calibration-cases", type=int)
    parser.add_argument(
        "--routing-mode",
        choices=(
            "original", "image", "joint", "joint_gate", "joint_empty_to_low"
        ),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train" and args.coverage_completion:
        if args.resume is None:
            parser.error("--coverage-completion requires --resume")
    config = apply_overrides(load_config(args.config), args)
    if (
        args.command == "train"
        and args.coverage_completion
        and str(config.get("dynamic_hard", {}).get("mode", "none"))
        != "dynamic_hard"
    ):
        parser.error("--coverage-completion is only valid for Dynamic-hard")
    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    logger = configure_logging(output_dir)
    seed_everything(int(config["experiment"]["seed"]))
    train_cases, test_cases = load_split(
        config["data"]["image_dir"],
        config["data"]["split_json"],
    )
    logger.info(
        "Loaded target split: train=%d test=%d", len(train_cases), len(test_cases)
    )
    prompt_cache_dir = Path(
        config.get("liver_adaptation", {}).get(
            "prompt_cache_dir", output_dir / "prompt_cache"
        )
    )
    runtime = build_runtime(config, prompt_cache_dir)
    trainable = sum(parameter.numel() for parameter in runtime.trainable_parameters)
    total = sum(parameter.numel() for parameter in runtime.model.parameters())
    logger.info(
        "VoxTell ready: LoRA QKV modules=%d trainable=%d total=%d device=%s",
        runtime.lora_modules,
        trainable,
        total,
        runtime.device,
    )
    _save_resolved_config(config, output_dir)
    if args.command == "train":
        validation_cases = [] if args.skip_eval else test_cases
        validation_gt_path = None if args.skip_eval else args.gt_path
        if validation_cases and validation_gt_path is None:
            raise ValueError("--gt-path is required for periodic validation")
        trainer = CUETrainer(
            config,
            runtime,
            train_cases,
            output_dir,
            logger,
            validation_cases=validation_cases,
            validation_gt_path=validation_gt_path,
        )
        checkpoint_path = trainer.run(
            resume=args.resume,
            confidence_partition=args.confidence_partition,
            coverage_completion=args.coverage_completion,
        )
        if args.skip_eval:
            return
        logger.info("Training complete; latest checkpoint: %s", checkpoint_path)
        return
    if args.checkpoint is not None:
        load_checkpoint(args.checkpoint, runtime.model)
        logger.info("Loaded LoRA checkpoint: %s", args.checkpoint)
    else:
        logger.info("Evaluating frozen VoxTell baseline (zero LoRA deltas)")
    evaluate(
        config,
        runtime,
        test_cases,
        args.gt_path,
        output_dir / "evaluation",
        logger,
    )


def _save_resolved_config(config: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
