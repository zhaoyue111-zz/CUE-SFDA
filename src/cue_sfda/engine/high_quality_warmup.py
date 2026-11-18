from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from cue_sfda.data import CaseRecord
from cue_sfda.engine.checkpoint import save_checkpoint
from cue_sfda.engine.trainer import CUETrainer
from cue_sfda.evaluation import evaluate
from cue_sfda.utils import save_json


EVALUATION_EPOCHS = (0, 5, 10, 15, 20)
WARMUP_EPOCHS = 20


def load_and_validate_routing(
    routing_csv: str | Path,
    cases: Iterable[CaseRecord],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    path = Path(routing_csv).expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "case_id" not in rows[0] or "assigned_branch" not in rows[0]:
        raise ValueError(f"Invalid or empty routing CSV: {path}")
    case_ids = [str(case.name) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Training split contains duplicate case IDs")
    route_ids = [str(row["case_id"]) for row in rows]
    duplicates = sorted(case_id for case_id, count in Counter(route_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Routing CSV has duplicate case IDs: {duplicates[:5]}")
    expected = set(case_ids)
    actual = set(route_ids)
    if actual != expected:
        raise ValueError(
            "Routing CSV does not exactly match the train split; "
            f"missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )
    groups = {str(row["case_id"]): str(row["assigned_branch"]).lower() for row in rows}
    invalid_groups = sorted(set(groups.values()) - {"high", "low"})
    if invalid_groups:
        raise ValueError(f"Warm-up routing only supports High/Low, got {invalid_groups}")
    high = {case_id for case_id, group in groups.items() if group == "high"}
    low = {case_id for case_id, group in groups.items() if group == "low"}
    if high & low or high | low != expected:
        raise ValueError("High and Low must be disjoint and cover the train split")
    no_response = {
        str(row["case_id"])
        for row in rows
        if str(row.get("response_state", "")).lower() == "no-response"
    }
    if no_response - low:
        raise ValueError(
            "The selected route is not the requested empty-prediction-to-Low split: "
            f"No-response cases assigned High={sorted(no_response - low)[:5]}"
        )
    return groups, rows


def shuffled_epoch_cases(
    cases: list[CaseRecord], rng: random.Random
) -> list[CaseRecord]:
    """Return one full epoch order, with every case appearing exactly once."""
    ordered = list(cases)
    rng.shuffle(ordered)
    return ordered


def summarize_epoch_metrics(
    epoch: int,
    per_case_dice: dict[str, float],
    groups: dict[str, str],
    baseline: dict[str, float] | None,
) -> tuple[dict[str, float | int], dict[str, float]]:
    expected = set(groups)
    if set(per_case_dice) != expected:
        raise ValueError(
            "Evaluation cases differ from saved routing: "
            f"missing={sorted(expected - set(per_case_dice))[:5]}, "
            f"extra={sorted(set(per_case_dice) - expected)[:5]}"
        )
    values = {key: float(value) for key, value in per_case_dice.items()}
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("Evaluation contains a non-finite Dice value")
    result: dict[str, float | int] = {"epoch": int(epoch)}
    for label, predicate in (
        ("all", lambda _: True),
        ("high", lambda case_id: groups[case_id] == "high"),
        ("low", lambda case_id: groups[case_id] == "low"),
    ):
        selected = [value for case_id, value in values.items() if predicate(case_id)]
        if not selected:
            raise ValueError(f"Cannot summarize empty {label} group")
        result[f"{label}_mean_dice"] = float(np.mean(selected))
    for label in ("all", "high", "low"):
        result[f"delta_{label}"] = (
            0.0
            if baseline is None
            else float(result[f"{label}_mean_dice"]) - float(baseline[f"{label}_mean_dice"])
        )
    return result, values


def should_replace_best(
    candidate_dice: float,
    candidate_epoch: int,
    best_dice: float | None,
    best_epoch: int | None,
) -> bool:
    if best_dice is None or best_epoch is None:
        return True
    return candidate_dice > best_dice or (
        candidate_dice == best_dice and candidate_epoch < best_epoch
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_resolved_config(config: dict, output_dir: Path) -> None:
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)


def _save_warmup_checkpoint(
    path: Path,
    trainer: CUETrainer,
    epoch: int,
    best_dice: float | None,
    best_epoch: int | None,
    epoch_rng: random.Random,
    trainer_state: dict,
) -> Path:
    state = {
        **trainer_state,
        "warmup_epoch": int(epoch),
        "epoch_python_rng_state": epoch_rng.getstate(),
        "train_group": "high_only",
    }
    return save_checkpoint(
        path=path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        iteration=epoch,
        config=trainer.config,
        confidence_scores=trainer.partition.scores,
        confidence_threshold=trainer.partition.threshold,
        best_dice=best_dice,
        best_iteration=best_epoch,
        trainer_generator_state=trainer.generator.get_state(),
        view_generator_state=trainer.views.generator.get_state(),
        view_python_generator_state=trainer.views.python_generator.getstate(),
        scaler=trainer.scaler,
        trainer_state=state,
    )


def run_high_quality_warmup(
    config: dict,
    runtime,
    cases: list[CaseRecord],
    gt_path: str | Path,
    output_dir: str | Path,
    logger,
) -> dict:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_route = Path(config["warmup"]["routing_csv"]).expanduser().resolve()
    source_checkpoint = Path(config["warmup"]["source_checkpoint"]).expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Routing-source checkpoint not found: {source_checkpoint}")
    groups, route_rows = load_and_validate_routing(source_route, cases)
    high_cases = [case for case in cases if groups[case.name] == "high"]
    low_ids = sorted(case.name for case in cases if groups[case.name] == "low")
    high_ids = sorted(case.name for case in high_cases)
    if len(cases) != 101 or len(high_cases) != 59 or len(low_ids) != 42:
        raise ValueError(
            "Expected the saved 101-case High=59/Low=42 routing; "
            f"got cases={len(cases)}, High={len(high_cases)}, Low={len(low_ids)}"
        )
    shutil.copy2(source_route, output_root / "routing.csv")
    source_note = {
        "routing_source": str(source_route),
        "routing_source_checkpoint_reference_only": str(source_checkpoint),
        "source_checkpoint_loaded": False,
        "source_checkpoint_validation_mean_dice": 0.3407987316410043,
        "task_note_reported_source_dice": 0.3404,
        "source_dice_discrepancy": (
            "The supplied checkpoint filename reports 0.340799; its adjacent "
            "validation summary records 0.3407987316410043 over 44 cases. The "
            "saved routing.csv is used as requested; checkpoint weights are not loaded."
        ),
        "high_count": len(high_ids),
        "low_count": len(low_ids),
        "skip_count": 0,
        "train_count": len(cases),
        "high_case_ids": high_ids,
        "low_case_ids": low_ids,
        "high_low_disjoint": not (set(high_ids) & set(low_ids)),
        "high_low_union_equals_train": set(high_ids) | set(low_ids) == {case.name for case in cases},
        "no_response_all_low": all(
            groups[str(row["case_id"])] == "low"
            for row in route_rows
            if str(row.get("response_state", "")).lower() == "no-response"
        ),
    }
    save_json(source_note, output_root / "split_summary.json")
    logger.info(
        "High-only warm-up routing source=%s checkpoint_reference=%s High=%d Low=%d Skip=0\n"
        "High IDs: %s\nLow IDs: %s",
        source_route,
        source_checkpoint,
        len(high_ids),
        len(low_ids),
        ",".join(high_ids),
        ",".join(low_ids),
    )
    _write_resolved_config(config, output_root)

    trainer = CUETrainer(config, runtime, cases, output_root, logger)
    trainer.partition.initialize(
        {case.name: (1.0 if groups[case.name] == "high" else 0.0) for case in cases}
    )
    trainer.model.train()
    epoch_rng = random.Random(int(config["experiment"]["seed"]))
    metric_rows: list[dict] = []
    per_case_rows: list[dict] = []
    training_rows: list[dict] = []
    baseline: dict[str, float] | None = None
    best_dice: float | None = None
    best_epoch: int | None = None
    best_summary: dict | None = None
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(epoch: int) -> dict:
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
        metrics_path = output_root / "evaluations" / f"epoch_{epoch:03d}" / "metrics_per_case.csv"
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        dice = {str(row["case"]): float(row["dice"]) for row in raw_rows}
        summary, values = summarize_epoch_metrics(epoch, dice, groups, baseline)
        metric_rows.append(summary)
        for case_id in sorted(values):
            per_case_rows.append(
                {"epoch": epoch, "case_id": case_id, "group": groups[case_id], "dice": values[case_id]}
            )
        logger.info(
            "epoch=%d All=%.6f High=%.6f Low=%.6f dAll=%+.6f dHigh=%+.6f dLow=%+.6f",
            epoch,
            summary["all_mean_dice"],
            summary["high_mean_dice"],
            summary["low_mean_dice"],
            summary["delta_all"],
            summary["delta_high"],
            summary["delta_low"],
        )
        return {"summary": summary, "per_case": values}

    baseline_result = run_evaluation(0)
    baseline = baseline_result["summary"]
    for epoch in range(1, WARMUP_EPOCHS + 1):
        trainer.model.train()
        epoch_losses = []
        effective_patches = 0
        skipped_patches = 0
        optimizer_steps = 0
        for case in shuffled_epoch_cases(high_cases, epoch_rng):
            scale_before = float(trainer.scaler.get_scale())
            metrics = trainer._train_step(case, case)
            scale_after = float(trainer.scaler.get_scale())
            epoch_losses.append(float(metrics["loss"]))
            effective_patches += int(metrics["effective_patches"])
            skipped_patches += int(metrics["skipped_patches"])
            amp_step_skipped = trainer.amp_enabled and scale_after < scale_before
            optimizer_steps += int(
                int(metrics["effective_patches"]) > 0 and not amp_step_skipped
            )
            trainer.last_training_iteration += 1
        training_row = {
            "epoch": epoch,
            "primary_cases": len(high_cases),
            "mean_loss": float(np.mean(epoch_losses)),
            "effective_patches": effective_patches,
            "skipped_patches": skipped_patches,
            "optimizer_steps": optimizer_steps,
        }
        training_rows.append(training_row)
        logger.info(
            "warmup_epoch=%d/%d primary_cases=%d mean_loss=%.6f effective_patches=%d "
            "skipped_patches=%d optimizer_steps=%d",
            epoch,
            WARMUP_EPOCHS,
            len(high_cases),
            training_row["mean_loss"],
            effective_patches,
            skipped_patches,
            optimizer_steps,
        )
        if epoch in EVALUATION_EPOCHS[1:]:
            result = run_evaluation(epoch)
            candidate = float(result["summary"]["high_mean_dice"])
            if should_replace_best(candidate, epoch, best_dice, best_epoch):
                best_dice = candidate
                best_epoch = epoch
                best_summary = result["summary"]
                _save_warmup_checkpoint(
                    checkpoint_dir / "best_high.pt",
                    trainer,
                    epoch,
                    best_dice,
                    best_epoch,
                    epoch_rng,
                    {"epoch_optimizer_steps": optimizer_steps},
                )

    last_path = _save_warmup_checkpoint(
        checkpoint_dir / "last.pt",
        trainer,
        WARMUP_EPOCHS,
        best_dice,
        best_epoch,
        epoch_rng,
        {"epoch_optimizer_steps": training_rows[-1]["optimizer_steps"]},
    )
    _write_csv(
        output_root / "training_log.csv",
        training_rows,
        ["epoch", "primary_cases", "mean_loss", "effective_patches", "skipped_patches", "optimizer_steps"],
    )
    _write_csv(
        output_root / "metrics_summary.csv",
        metric_rows,
        ["epoch", "all_mean_dice", "high_mean_dice", "low_mean_dice", "delta_all", "delta_high", "delta_low"],
    )
    _write_csv(
        output_root / "metrics_per_case.csv",
        per_case_rows,
        ["epoch", "case_id", "group", "dice"],
    )
    if best_summary is None or best_epoch is None or best_dice is None:
        raise RuntimeError("No eligible High Dice checkpoint was selected")
    baseline_cases = baseline_result["per_case"]
    best_case_rows = [row for row in per_case_rows if int(row["epoch"]) == best_epoch]
    changes = []
    for row in best_case_rows:
        case_id = str(row["case_id"])
        changes.append(
            {
                **row,
                "delta_dice": float(row["dice"]) - baseline_cases[case_id],
                "baseline_dice": baseline_cases[case_id],
            }
        )
    group_changes = {}
    for group in ("high", "low"):
        selected = [row for row in changes if row["group"] == group]
        group_changes[group] = {
            "improved": sum(row["delta_dice"] > 0 for row in selected),
            "degraded": sum(row["delta_dice"] < 0 for row in selected),
            "unchanged": sum(row["delta_dice"] == 0 for row in selected),
            "largest_improvements": sorted(selected, key=lambda row: row["delta_dice"], reverse=True)[:5],
            "largest_degradations": sorted(selected, key=lambda row: row["delta_dice"])[:5],
        }
    best_report = {
        "selection_metric": "High mean real Dice on training GT (oracle diagnostic)",
        "eligible_epochs": list(EVALUATION_EPOCHS[1:]),
        "best_epoch": best_epoch,
        "best_high_mean_dice": best_dice,
        "best_metrics": best_summary,
        "best_checkpoint": str(checkpoint_dir / "best_high.pt"),
        "last_epoch": WARMUP_EPOCHS,
        "last_checkpoint": str(last_path),
        "group_case_change_counts_and_extremes": group_changes,
    }
    save_json(best_report, output_root / "best_checkpoint_summary.json")
    logger.info("High-only Warm-up complete; selected epoch=%d High Dice=%.6f", best_epoch, best_dice)
    return best_report
