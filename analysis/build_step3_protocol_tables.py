#!/usr/bin/env python3
"""Summarize existing low-branch experiment configs and 44-case results."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = ROOT / "outputs/ablation"
OLD_SUMMARY = ABLATION_ROOT / "summary/ablation_results.csv"
METHODS = [
    ("C_joint", ROOT / "outputs/response_routing/C_joint", "C_joint"),
    ("Image-only", ABLATION_ROOT / "B_image", "B_image"),
    ("Prompt-only", ABLATION_ROOT / "C_prompt", "C_prompt"),
    ("Cross-modal", ABLATION_ROOT / "D_cross_modal", "D_cross_modal"),
    ("Cross-modal+Language", ABLATION_ROOT / "E_language", "E_language"),
    ("Dynamic-hard", ABLATION_ROOT / "G_dynamic_hard_final", "G_dynamic_hard_final"),
    ("Dynamic-EMA", ABLATION_ROOT / "H_dynamic_ema_final", "H_dynamic_ema_final"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metric_row(root: Path, name: str) -> tuple[dict, dict, dict]:
    config = read_json(root / "resolved_config.json")
    early = read_json(root / "early_stopping.json")
    if name == "C_joint":
        best_name = next((root / "checkpoints").glob("best_iteration_*_dice_*.pt"), None)
        match = re.search(r"best_iteration_(\d+)_", best_name.name) if best_name else None
        best_iter = int(match.group(1)) if match else ""
        stop_iter = 1000  # resolved iteration cap; this run has no early-stopping state file.
        metric_path = root / "best_evaluation/evaluation/metrics_summary.csv"
        metrics = read_rows(metric_path)[0] if metric_path.exists() else {}
        checkpoint = str(best_name.relative_to(ROOT)) if best_name else ""
        refresh = {}
    else:
        best_iter = early.get("best_iteration", "")
        stop_iter = early.get("stop_iteration", "")
        files = list((root / "checkpoints").glob(f"best_iteration_{int(best_iter):06d}_dice_*.pt")) if best_iter != "" else []
        checkpoint = str(files[0].relative_to(ROOT)) if files else ""
        metric_path = root / "validation" / f"iteration_{int(best_iter):06d}" / "metrics_summary.csv" if best_iter != "" else Path("/no/file")
        metrics = read_rows(metric_path)[0] if metric_path.exists() else {}
        refresh = read_json(root / "refresh_history.json")
    return config, {"best_iter": best_iter, "stop_iter": stop_iter, "checkpoint": checkpoint, "metrics": metrics}, refresh


def main() -> None:
    out = ROOT / "analysis/tables"
    cjoint_route = {r["case_id"]: r["assigned_branch"] for r in read_rows(ROOT / "outputs/response_routing/C_joint/routing.csv")}
    audit = []
    overall = []
    route_identity = []
    static_names = {"Image-only", "Prompt-only", "Cross-modal", "Cross-modal+Language"}
    for label, root, key in METHODS:
        config, result, refresh = metric_row(root, label)
        method = config.get("method", {})
        consensus = config.get("consensus", {})
        stage2 = config.get("stage2", {})
        routing_mode = method.get("routing_mode", "not recorded")
        consensus_mode = consensus.get("mode", "none")
        stage2_mode = stage2.get("mode", "none (default)")
        p_high = method.get("high_quality_sampling_probability", "")
        rsummary = refresh.get("refresh_history", [])
        refresh_iters = ";".join(str(x.get("trigger_iteration", "")) for x in rsummary)
        if label in {"Dynamic-hard", "Dynamic-EMA"}:
            ordinary_until = 550 if label == "Dynamic-hard" else 500
            complete_at = 653 if label == "Dynamic-hard" else 603
            sampler = (
                f"hybrid, reconstructed by seeded RNG replay: normal p_high={p_high} through {ordinary_until}; "
                f"unseen-primary coverage fills last 2 cases at {ordinary_until+1}/{ordinary_until+2}; "
                f"after refresh coverage restarts and completes by {complete_at}, then p_high resumes"
            )
            sampler_evidence = "logged pre-refresh samples match ordinary seeded sampler; resumed post-refresh sample IDs match unseen sampler replay"
        elif label in static_names:
            sampler = f"ordinary branch sampler p_high={p_high}; no unseen-priority observed"
            sampler_evidence = "iteration 10 and 100 repeat the same primary case; no refresh"
        else:
            sampler = f"ordinary branch sampler p_high={p_high}; consensus disabled"
            sampler_evidence = "routing-only C_joint baseline; no consensus refresh state"

        routing_path = root / "routing.csv"
        route_diff = ""
        route_count = ""
        if routing_path.exists():
            local = {r["case_id"]: r["assigned_branch"] for r in read_rows(routing_path)}
            route_count = len(local)
            route_diff = sum(local.get(case) != branch for case, branch in cjoint_route.items())
            route_identity.append({
                "method": label,
                "routing_mode": routing_mode,
                "routing_cases": len(local),
                "mismatches_vs_fixed_C_joint": route_diff,
                "partition_source": str(routing_path.relative_to(ROOT)),
            })
        else:
            route_identity.append({"method": label, "routing_mode": routing_mode, "routing_cases": "", "mismatches_vs_fixed_C_joint": "", "partition_source": "not present"})
        audit.append({
            "method": label,
            "artifact": str(root.relative_to(ROOT)),
            "routing_mode": routing_mode,
            "consensus_mode": consensus_mode,
            "stage2_mode": stage2_mode,
            "stage2_key_explicit": "stage2" in config,
            "high_quality_sampling_probability": p_high,
            "best_iteration": result["best_iter"],
            "stop_iteration": result["stop_iter"],
            "best_checkpoint": result["checkpoint"],
            "test_case_count": 44 if result["metrics"] else "",
            "overall_dice": result["metrics"].get("mean_dice", ""),
            "overall_miou": result["metrics"].get("mean_iou", ""),
            "overall_precision": result["metrics"].get("mean_precision", ""),
            "overall_recall": result["metrics"].get("mean_recall", ""),
            "refresh_count": len(rsummary),
            "refresh_iterations": refresh_iters,
            "cache_ema_beta": ";".join(str(x.get("cache_ema_beta", "")) for x in rsummary),
            "legacy_sampler_protocol": sampler,
            "sampler_evidence": sampler_evidence,
            "partition_case_count": route_count,
            "partition_mismatches_vs_fixed_C_joint": route_diff,
        })
        metrics = result["metrics"]
        overall.append({
            "method": label,
            "protocol_family": "historical response-routing/ablation (not liver_cancer_mainline)",
            "best_iteration": result["best_iter"],
            "stop_iteration": result["stop_iter"],
            "test_cases": 44 if metrics else "",
            "dice": metrics.get("mean_dice", ""),
            "mIoU": metrics.get("mean_iou", ""),
            "precision": metrics.get("mean_precision", ""),
            "recall": metrics.get("mean_recall", ""),
            "note": "44-case validation/test summary only; not train High/Low metrics",
        })

    write(out / "step3_protocol_audit.csv", audit)
    write(out / "step3_overall_existing_results.csv", overall)
    write(out / "step3_route_identity_audit.csv", route_identity)
    write(out / "step3_train_group_data_availability.csv", [{
        "required_dataset": "fixed C_joint training partition",
        "train_cases": 101,
        "frozen_pre_adaptation_dice_rows": 101,
        "frozen_pre_adaptation_precision_recall_available": False,
        "adapted_per_case_rows_on_train_cases": 0,
        "available_adapted_per_case_rows_are_test_cases": 44,
        "high_low_pre_post_analysis_possible": False,
        "reason": "existing adapted metrics_per_case.csv are for the 44 held-out validation/test cases; no per-case predictions or metrics for the 101 training cases",
    }])
    print("wrote", out)


if __name__ == "__main__":
    main()
