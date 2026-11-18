from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from cue_sfda.data import CaseRecord
from cue_sfda.engine.checkpoint import save_checkpoint
from cue_sfda.engine.high_quality_warmup import (
    load_and_validate_routing,
    shuffled_epoch_cases,
)
from cue_sfda.engine.trainer import CUETrainer
from cue_sfda.evaluation import evaluate
from cue_sfda.utils import save_json


METRICS = ("dice", "iou", "precision", "recall")


def _read_per_case(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row["case"]): {metric: float(row[metric]) for metric in METRICS}
        for row in rows
    }


def _summarize_metrics(
    epoch: int,
    values: dict[str, dict[str, float]],
    groups: dict[str, str],
    baseline: dict[str, dict[str, float]] | None,
) -> tuple[list[dict], list[dict]]:
    expected = set(groups)
    if set(values) != expected:
        raise ValueError(
            "Evaluation cases differ from saved routing: "
            f"missing={sorted(expected - set(values))[:5]}, "
            f"extra={sorted(set(values) - expected)[:5]}"
        )
    if any(not np.isfinite(metric) for row in values.values() for metric in row.values()):
        raise ValueError("Evaluation contains non-finite metrics")
    summaries = []
    case_rows = []
    for group in ("all", "high", "low"):
        selected = sorted(
            case_id for case_id in values
            if group == "all" or groups[case_id] == group
        )
        if not selected:
            raise ValueError(f"Cannot summarize empty {group} group")
        summary = {"epoch": int(epoch), "group": group, "case_count": len(selected)}
        for metric in METRICS:
            mean = float(np.mean([values[case_id][metric] for case_id in selected]))
            summary[f"mean_{metric}"] = mean
            summary[f"delta_{metric}"] = (
                0.0
                if baseline is None
                else mean - float(np.mean([baseline[case_id][metric] for case_id in selected]))
            )
        summaries.append(summary)
    for case_id in sorted(values):
        row = {"epoch": int(epoch), "case_id": case_id, "group": groups[case_id]}
        for metric in METRICS:
            row[metric] = values[case_id][metric]
            row[f"delta_{metric}"] = (
                0.0 if baseline is None else values[case_id][metric] - baseline[case_id][metric]
            )
        case_rows.append(row)
    return summaries, case_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_warmup_checkpoint(
    path: Path,
    trainer: CUETrainer,
    iteration: int,
    epoch: int,
    best_dice: float | None,
    best_epoch: int | None,
    epoch_rng: random.Random,
) -> Path:
    trainer_state = {
        **trainer._trainer_checkpoint_state(),
        "warmup_epoch": int(epoch),
        "epoch_python_rng_state": epoch_rng.getstate(),
        "train_group": "low_only",
        "best_epoch": best_epoch,
    }
    return save_checkpoint(
        path=path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        iteration=iteration,
        config=trainer.config,
        confidence_scores=trainer.partition.scores,
        confidence_threshold=trainer.partition.threshold,
        best_dice=best_dice,
        best_iteration=trainer.best_iteration,
        trainer_generator_state=trainer.generator.get_state(),
        view_generator_state=trainer.views.generator.get_state(),
        view_python_generator_state=trainer.views.python_generator.getstate(),
        scaler=trainer.scaler,
        trainer_state=trainer_state,
    )


def run_low_only_dynamic_hard_warmup(
    config: dict,
    runtime,
    cases: list[CaseRecord],
    gt_path: str | Path,
    output_dir: str | Path,
    logger,
) -> dict:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    groups, route_rows = load_and_validate_routing(
        config["low_only_warmup"]["routing_csv"], cases
    )
    high_cases = [case for case in cases if groups[case.name] == "high"]
    low_cases = [case for case in cases if groups[case.name] == "low"]
    if len(cases) != 101 or len(high_cases) != 59 or len(low_cases) != 42:
        raise ValueError(
            "Expected saved M1 route with 101 cases, High=59, Low=42; got "
            f"{len(cases)}, {len(high_cases)}, {len(low_cases)}"
        )
    shutil.copy2(
        Path(config["low_only_warmup"]["routing_csv"]).expanduser().resolve(),
        output_root / "routing.csv",
    )
    save_json(
        {
            "routing_source": str(Path(config["low_only_warmup"]["routing_csv"]).resolve()),
            "high_count": len(high_cases),
            "low_count": len(low_cases),
            "skip_count": 0,
            "train_count": len(cases),
            "high_case_ids": sorted(case.name for case in high_cases),
            "low_case_ids": sorted(case.name for case in low_cases),
            "training_cases": "Low only",
            "dynamic_hard_cache_cases": len(low_cases),
            "dynamic_hard_refresh_coverage": "all Low cases, at most once",
        },
        output_root / "split_summary.json",
    )
    logger.info(
        "Low-only Dynamic-hard warm-up: High=%d Low=%d; only Low updates; "
        "Low IDs=%s",
        len(high_cases),
        len(low_cases),
        ",".join(sorted(case.name for case in low_cases)),
    )
    with (output_root / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)

    epochs = int(config["low_only_warmup"]["epochs"])
    eval_epochs = tuple(int(value) for value in config["low_only_warmup"]["evaluation_epochs"])
    if epochs != 20 or eval_epochs != (0, 5, 10, 15, 20):
        raise ValueError("This experiment requires 20 epochs and evaluations at 0/5/10/15/20")
    trainer = CUETrainer(config, runtime, low_cases, output_root, logger)
    trainer.partition.initialize({case.name: 0.0 for case in low_cases})
    trainer._ensure_dynamic_hard_cache()
    trainer.model.train()
    epoch_rng = random.Random(int(config["experiment"]["seed"]))
    step_count = 0
    training_rows = []
    summary_rows = []
    per_case_rows = []
    baseline = None
    best_high_dice = None
    best_epoch = None
    best_path = output_root / "checkpoints" / "best_high.pt"
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_epoch(epoch: int) -> dict[str, dict[str, float]]:
        was_training = trainer.model.training
        try:
            evaluate(
                config,
                runtime,
                cases,
                gt_path,
                output_root / "evaluations" / f"epoch_{epoch:03d}",
                logger,
                save_predictions=False,
            )
        finally:
            trainer.model.train(was_training)
        values = _read_per_case(
            output_root / "evaluations" / f"epoch_{epoch:03d}" / "metrics_per_case.csv"
        )
        summaries, case_rows = _summarize_metrics(epoch, values, groups, baseline)
        summary_rows.extend(summaries)
        per_case_rows.extend(case_rows)
        for row in summaries:
            logger.info(
                "epoch=%d group=%s n=%d Dice=%.6f mIoU=%.6f Precision=%.6f Recall=%.6f",
                epoch,
                row["group"],
                row["case_count"],
                row["mean_dice"],
                row["mean_iou"],
                row["mean_precision"],
                row["mean_recall"],
            )
        return values

    baseline = evaluate_epoch(0)
    for epoch in range(1, epochs + 1):
        trainer.model.train()
        losses = []
        effective_patches = 0
        skipped_patches = 0
        optimizer_steps = 0
        for case in shuffled_epoch_cases(low_cases, epoch_rng):
            partner_index = int(torch.randint(len(cases), (1,), generator=trainer.generator).item())
            scale_before = float(trainer.scaler.get_scale())
            metrics = trainer._train_step(case, cases[partner_index])
            scale_after = float(trainer.scaler.get_scale())
            step_count += 1
            trainer.last_training_iteration = step_count
            trainer.branch_sample_counts["low"] += 1
            trainer.branch_patch_counts["low"] += int(metrics["patches"])
            trainer.branch_effective_patch_counts["low"] += int(metrics["effective_patches"])
            trainer.branch_skipped_patch_counts["low"] += int(metrics["skipped_patches"])
            trainer.dynamic_hard_loss_total += float(metrics.get("dynamic_hard_loss", 0.0)) * int(metrics["patches"])
            trainer.dynamic_hard_loss_patches += int(metrics["patches"])
            losses.append(float(metrics["loss"]))
            effective_patches += int(metrics["effective_patches"])
            skipped_patches += int(metrics["skipped_patches"])
            amp_step_skipped = trainer.amp_enabled and scale_after < scale_before
            optimizer_steps += int(int(metrics["effective_patches"]) > 0 and not amp_step_skipped)
            trainer.seen_cases_since_refresh.add(case.name)
            if trainer._dynamic_hard_refresh_ready():
                trainer._refresh_dynamic_hard_cache(step_count)
                trainer._save_latest(step_count)
        training_rows.append(
            {
                "epoch": epoch,
                "primary_cases": len(low_cases),
                "mean_loss": float(np.mean(losses)),
                "effective_patches": effective_patches,
                "skipped_patches": skipped_patches,
                "optimizer_steps": optimizer_steps,
                "cache_version": trainer.refresh_cycle,
                "refresh_count": trainer.refresh_cycle,
            }
        )
        logger.info(
            "low_only_epoch=%d/%d cases=%d mean_loss=%.6f effective_patches=%d "
            "skipped_patches=%d optimizer_steps=%d cache_version=%d",
            epoch,
            epochs,
            len(low_cases),
            training_rows[-1]["mean_loss"],
            effective_patches,
            skipped_patches,
            optimizer_steps,
            trainer.refresh_cycle,
        )
        if epoch in eval_epochs[1:]:
            current = evaluate_epoch(epoch)
            high_dice = float(np.mean([current[case.name]["dice"] for case in high_cases]))
            if best_high_dice is None or high_dice > best_high_dice:
                best_high_dice = high_dice
                best_epoch = epoch
                trainer.best_dice = best_high_dice
                trainer.best_iteration = step_count
                trainer.best_checkpoint_path = best_path
                _save_warmup_checkpoint(
                    best_path,
                    trainer,
                    step_count,
                    epoch,
                    best_high_dice,
                    best_epoch,
                    epoch_rng,
                )

    last_path = _save_warmup_checkpoint(
        checkpoint_dir / "last.pt",
        trainer,
        step_count,
        epochs,
        best_high_dice,
        best_epoch,
        epoch_rng,
    )
    trainer._write_training_summary()
    _write_csv(output_root / "training_log.csv", training_rows)
    _write_csv(output_root / "metrics_summary.csv", summary_rows)
    _write_csv(output_root / "metrics_per_case.csv", per_case_rows)
    report = {
        "best_epoch": best_epoch,
        "best_high_mean_dice": best_high_dice,
        "selection_metric": "High mean Dice at evaluations 5/10/15/20 (oracle diagnostic)",
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "dynamic_hard_refresh_history": trainer.refresh_history,
        "dynamic_hard_cache_version": trainer.refresh_cycle,
    }
    best_case_rows = [row for row in per_case_rows if int(row["epoch"]) == best_epoch]
    changes = [
        {
            **row,
            "delta_dice_from_epoch0": float(row["dice"]) - baseline[row["case_id"]]["dice"],
        }
        for row in best_case_rows
    ]
    report["best_checkpoint_case_dice_changes"] = {
        group: {
            "improved": sum(
                row["delta_dice_from_epoch0"] > 0
                for row in changes if row["group"] == group
            ),
            "degraded": sum(
                row["delta_dice_from_epoch0"] < 0
                for row in changes if row["group"] == group
            ),
            "unchanged": sum(
                row["delta_dice_from_epoch0"] == 0
                for row in changes if row["group"] == group
            ),
            "largest_improvements": sorted(
                (row for row in changes if row["group"] == group),
                key=lambda row: row["delta_dice_from_epoch0"],
                reverse=True,
            )[:5],
            "largest_degradations": sorted(
                (row for row in changes if row["group"] == group),
                key=lambda row: row["delta_dice_from_epoch0"],
            )[:5],
        }
        for group in ("high", "low")
    }
    save_json(report, output_root / "best_checkpoint_summary.json")
    return report
