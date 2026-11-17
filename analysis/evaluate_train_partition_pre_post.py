#!/usr/bin/env python3
"""Evaluate frozen VoxTell and saved LoRA checkpoints on the 101 train cases.

This script performs inference only. It never constructs an optimizer, calls a
training step, or writes prediction masks. It is intentionally not run as part
of the current read-only analysis because the requested train-case 3D outputs
are not already available.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cue_sfda.config import load_config  # noqa: E402
from cue_sfda.data import load_split  # noqa: E402
from cue_sfda.engine.checkpoint import load_checkpoint  # noqa: E402
from cue_sfda.evaluation import evaluate  # noqa: E402
from cue_sfda.modeling import build_runtime  # noqa: E402
from cue_sfda.utils import configure_logging, seed_everything  # noqa: E402

DEFAULT_CONFIG = ROOT / "outputs/response_routing/C_joint/resolved_config.json"
DEFAULT_ROUTING = ROOT / "outputs/response_routing/C_joint/routing.csv"
def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=CHECKPOINT_PATH")
    return name, Path(path).expanduser().resolve()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def evaluate_once(config: dict, cases: list, checkpoint: Path | None, label: str, gt_path: Path, out_dir: Path) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir)
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    runtime = build_runtime(config, out_dir / "prompt_cache")
    if checkpoint is not None:
        load_checkpoint(checkpoint, runtime.model)
        logger.info("Loaded checkpoint for inference: %s", checkpoint)
    else:
        # Newly attached LoRA B matrices are zero-initialized, so this is the
        # unadapted VoxTell model. No training or optimizer is created here.
        logger.info("Evaluating frozen VoxTell with zero-initialized LoRA deltas")
    evaluate(
        config,
        runtime,
        cases,
        gt_path,
        out_dir / "metrics",
        logger,
        save_predictions=False,
    )
    result = {row["case"]: row for row in rows(out_dir / "metrics/metrics_per_case.csv")}
    del runtime
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/train_partition_eval")
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        help="Best checkpoint as NAME=PATH; repeat to evaluate additional methods.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # Resolved experiment config files contain a _config_path key but also all
    # resolved values. The model/data/evaluation fields are used unchanged.
    train_cases, _ = load_split(config["data"]["image_dir"], config["data"]["split_json"])
    if len(train_cases) != 101:
        raise ValueError(f"expected 101 train cases, found {len(train_cases)}")
    route = {row["case_id"]: row["assigned_branch"] for row in rows(args.routing)}
    case_names = {case.name for case in train_cases}
    if set(route) != case_names:
        raise ValueError("fixed C_joint routing does not exactly match the train split")
    if any(group not in {"high", "low"} for group in route.values()):
        raise ValueError("C_joint partition must contain only high/low; skip cases are unsupported")

    checkpoints = args.checkpoint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, dict]] = {}
    all_results["Frozen-VoxTell"] = evaluate_once(
        config, train_cases, None, "Frozen-VoxTell", args.gt_path, args.output_dir / "Frozen-VoxTell"
    )
    for name, checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        all_results[name] = evaluate_once(
            config, train_cases, checkpoint, name, args.gt_path, args.output_dir / name
        )

    baseline = all_results["Frozen-VoxTell"]
    per_case = []
    group_summary = []
    for name, current in all_results.items():
        if name == "Frozen-VoxTell":
            continue
        joined = []
        for case in sorted(case_names):
            if case not in baseline or case not in current:
                raise ValueError(f"missing metrics for {case} in {name}")
            pre, post = baseline[case], current[case]
            row = {
                "case_id": case,
                "fixed_C_joint_group": route[case],
                "method": name,
                "pre_dice": float(pre["dice"]),
                "post_dice": float(post["dice"]),
                "delta_dice": float(post["dice"]) - float(pre["dice"]),
                "pre_precision": float(pre["precision"]),
                "post_precision": float(post["precision"]),
                "delta_precision": float(post["precision"]) - float(pre["precision"]),
                "pre_recall": float(pre["recall"]),
                "post_recall": float(post["recall"]),
                "delta_recall": float(post["recall"]) - float(pre["recall"]),
                "pre_predicted_voxels": int(pre["predicted_voxels"]),
                "post_predicted_voxels": int(post["predicted_voxels"]),
                "ground_truth_voxels": int(post["ground_truth_voxels"]),
            }
            per_case.append(row)
            joined.append(row)
        for group in ("high", "low"):
            selected = [r for r in joined if r["fixed_C_joint_group"] == group]
            dd = [r["delta_dice"] for r in selected]
            dp = [r["delta_precision"] for r in selected]
            dr = [r["delta_recall"] for r in selected]
            group_summary.append({
                "method": name,
                "group": group,
                "n": len(selected),
                "pre_dice_mean": mean([r["pre_dice"] for r in selected]),
                "post_dice_mean": mean([r["post_dice"] for r in selected]),
                "delta_dice_mean": mean(dd),
                "delta_dice_median": median(dd),
                "improved_cases": sum(x > 0 for x in dd),
                "degraded_cases": sum(x < 0 for x in dd),
                "delta_dice_gt_0_02": sum(x > 0.02 for x in dd),
                "delta_dice_lt_minus_0_02": sum(x < -0.02 for x in dd),
                "pre_precision_mean": mean([r["pre_precision"] for r in selected]),
                "post_precision_mean": mean([r["post_precision"] for r in selected]),
                "delta_precision_mean": mean(dp),
                "pre_recall_mean": mean([r["pre_recall"] for r in selected]),
                "post_recall_mean": mean([r["post_recall"] for r in selected]),
                "delta_recall_mean": mean(dr),
            })
    write_csv(args.output_dir / "train_case_pre_post_per_case.csv", per_case)
    write_csv(args.output_dir / "train_case_pre_post_by_group.csv", group_summary)
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps({
            "protocol": "inference_only; no training; no prediction masks written",
            "train_cases": len(train_cases),
            "partition_source": str(args.routing.resolve()),
            "partition_groups": {g: sum(v == g for v in route.values()) for g in ("high", "low")},
            "frozen_model": "CUE-SFDA runtime with zero-initialized LoRA deltas",
            "checkpoints": {name: str(path.resolve()) for name, path in checkpoints},
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
