from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research_v2"
RUNS = {
    "E3_R0": "E3_eval",
    "E6_fixed50": "E6_eval",
    "E7_random50": "E7_eval_final",
    "E8_norm50": "E8_eval_safe_v2",
    "E9_cosine50": "E9_eval_safe_v2",
}
ROUTE_RUNS = (
    ("E6_fixed50_E3base", "E6_fixed50"),
    ("E7_random50_E3base", "E7_random50"),
    ("E8_norm50_safe_v2_E3base", "E8_norm50"),
    ("E9_cosine50_safe_v2_E3base", "E9_cosine50"),
)
METRICS = ("dice", "precision", "recall", "iou")


def read_metrics(run: str) -> dict[str, dict[str, float]]:
    path = OUT / run / "evaluation" / "metrics_per_case.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["case"]: {metric: float(row[metric]) for metric in METRICS}
            for row in csv.DictReader(handle)
        }


def main() -> None:
    results = {name: read_metrics(run) for name, run in RUNS.items()}
    cases = sorted(results["E3_R0"])
    if any(set(rows) != set(cases) for rows in results.values()):
        raise ValueError("Stage D evaluations must contain identical held-out cases")
    rng = np.random.default_rng(42)
    sampled = rng.integers(0, len(cases), size=(10000, len(cases)))
    deltas: dict[str, np.ndarray] = {}
    for name in results:
        deltas[name] = np.asarray([
            results[name][case]["dice"] - results["E3_R0"][case]["dice"]
            for case in cases
        ])

    analysis = OUT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    per_case_path = analysis / "per_case_metrics_stage_d.csv"
    fields = ["case_id"] + [f"{name}_{metric}" for name in results for metric in METRICS]
    fields += [f"{name}_delta_dice_vs_E3" for name in results if name != "E3_R0"]
    with per_case_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, case in enumerate(cases):
            row: dict[str, str | float] = {"case_id": case}
            for name, values in results.items():
                row.update({f"{name}_{metric}": values[case][metric] for metric in METRICS})
            for name in results:
                if name != "E3_R0":
                    row[f"{name}_delta_dice_vs_E3"] = deltas[name][index]
            writer.writerow(row)

    summary_rows = []
    for name, values in results.items():
        summary_rows.append({
            "run": name,
            **{f"mean_{metric}": float(np.mean([values[c][metric] for c in cases])) for metric in METRICS},
            **({} if name == "E3_R0" else {
                "mean_delta_dice_vs_E3": float(deltas[name].mean()),
                "median_delta_dice_vs_E3": float(np.median(deltas[name])),
                "delta_dice_ci95_low": float(np.quantile(deltas[name][sampled].mean(axis=1), 0.025)),
                "delta_dice_ci95_high": float(np.quantile(deltas[name][sampled].mean(axis=1), 0.975)),
                "improved_cases": int((deltas[name] > 0).sum()),
                "unchanged_cases": int((deltas[name] == 0).sum()),
                "declined_cases": int((deltas[name] < 0).sum()),
                "delta_gt_002": int((deltas[name] > 0.02).sum()),
                "delta_lt_minus_002": int((deltas[name] < -0.02).sum()),
            }),
        })
    summary_path = analysis / "summary_stage_d.csv"
    summary_fields = list(dict.fromkeys(key for row in summary_rows for key in row))
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    route_lines = ["| Run | Low updates | Mean selected groups/update | Group | Selected | Frequency | Mean finite score | Nonfinite scores |",
                   "|---|---:|---:|---|---:|---:|---:|---:|"]
    route_fields = ["run", "low_updates", "mean_selected_groups_per_update", "group", "selected_count", "selection_frequency", "mean_score", "nonfinite_score_count"]
    route_rows = []
    for run, label in ROUTE_RUNS:
        path = OUT / run / "parameter_routing.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            raw = list(csv.DictReader(handle))
        by_iteration: dict[str, list[dict[str, str]]] = {}
        for row in raw:
            by_iteration.setdefault(row["iteration"], []).append(row)
        low_updates = len(by_iteration)
        selected_per_update = [sum(row["selected"].lower() == "true" for row in rows) for rows in by_iteration.values()]
        mean_selected = float(np.mean(selected_per_update))
        for group in sorted({row["group"] for row in raw}):
            rows = [row for row in raw if row["group"] == group]
            selected = sum(row["selected"].lower() == "true" for row in rows)
            scores = np.asarray([float(row["score"]) for row in rows])
            finite_scores = scores[np.isfinite(scores)]
            mean_score = float(np.mean(finite_scores)) if finite_scores.size else float("nan")
            nonfinite_count = int((~np.isfinite(scores)).sum())
            route_rows.append({
                "run": label,
                "low_updates": low_updates,
                "mean_selected_groups_per_update": mean_selected,
                "group": group,
                "selected_count": selected,
                "selection_frequency": selected / max(len(rows), 1),
                "mean_score": mean_score,
                "nonfinite_score_count": nonfinite_count,
            })
            mean_display = f"{mean_score:.6f}" if np.isfinite(mean_score) else "—"
            route_lines.append(f"| {label} | {low_updates} | {mean_selected:.2f} | {group} | {selected} | {selected / max(len(rows), 1):.3f} | {mean_display} | {nonfinite_count} |")
    route_path = analysis / "routing_frequency_stage_d.csv"
    with route_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=route_fields)
        writer.writeheader()
        writer.writerows(route_rows)

    # Compare final adapter states with R0 to quantify end-of-training offsets.
    baseline = torch.load(OUT / "E3_raw_low_baseline" / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)["lora_state"]
    offset_fields = ["run", "group", "l2_offset_from_E3", "relative_l2_offset_from_E3"]
    offset_rows = []
    group_pattern = re.compile(r"layers\.(\d+)\.multihead_attn\.lora_([qkv])_([AB])$")
    for run, label in ROUTE_RUNS:
        state = torch.load(OUT / run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)["lora_state"]
        if set(state) != set(baseline):
            raise ValueError(f"Adapter keys differ between E3 and {run}")
        accum: dict[str, list[torch.Tensor]] = {}
        baseline_accum: dict[str, list[torch.Tensor]] = {}
        for key, value in state.items():
            match = group_pattern.search(key)
            if not match:
                continue
            layer, projection, _matrix = match.groups()
            group = f"layer{layer}_{projection.upper()}"
            accum.setdefault(group, []).append((value.float() - baseline[key].float()).reshape(-1))
            baseline_accum.setdefault(group, []).append(baseline[key].float().reshape(-1))
        for group in sorted(accum):
            offset_norm = float(torch.linalg.vector_norm(torch.cat(accum[group])).item())
            base_norm = float(torch.linalg.vector_norm(torch.cat(baseline_accum[group])).item())
            offset_rows.append({
                "run": label,
                "group": group,
                "l2_offset_from_E3": offset_norm,
                "relative_l2_offset_from_E3": offset_norm / max(base_norm, 1e-12),
            })
    offset_path = analysis / "parameter_offsets_stage_d.csv"
    with offset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=offset_fields)
        writer.writeheader()
        writer.writerows(offset_rows)

    lines = [
        "# Stage D: 50% decoder Q/K/V group routing",
        "",
        "Fixed-iteration 1000-step diagnostics on the previously explored 44-case held-out development set. It is not an untouched final test, and GT was not used during training or checkpoint choice.",
        "",
        "## Held-out performance",
        "",
        "| Run | Dice | Precision | Recall | IoU | Mean paired ΔDice vs E3 | Median ΔDice | 95% paired bootstrap CI | Improved / same / declined |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["run"] == "E3_R0":
            lines.append(f"| E3_R0 | {row['mean_dice']:.6f} | {row['mean_precision']:.6f} | {row['mean_recall']:.6f} | {row['mean_iou']:.6f} | — | — | — | — |")
        else:
            lines.append(
                f"| {row['run']} | {row['mean_dice']:.6f} | {row['mean_precision']:.6f} | {row['mean_recall']:.6f} | {row['mean_iou']:.6f} | "
                f"{row['mean_delta_dice_vs_E3']:.6f} | {row['median_delta_dice_vs_E3']:.6f} | "
                f"[{row['delta_dice_ci95_low']:.6f}, {row['delta_dice_ci95_high']:.6f}] | "
                f"{row['improved_cases']} / {row['unchanged_cases']} / {row['declined_cases']} |"
            )
    lines += [
        "",
        "## Stage conclusion",
        "",
        "None of the 50% routing strategies exceeded the E3/R0 all-group baseline. E9 cosine was the strongest sparse variant at Dice 0.302982, but remained 0.009535 below E3; its paired interval crosses zero and its Recall was 0.392321 versus E3's 0.439901. Fixed, random, and norm routing were lower still. The current evidence does not support dynamic routing as a standalone improvement; keep E3 as the reference for later combination diagnostics and do not promote E9 as a selected method.",
        "",
        "The bootstrap resamples the 44 paired case-level Dice differences (10,000 replicates, seed 42). These intervals describe this reused development set and do not replace seed replication.",
        "",
        "## Parameter routing audit",
        "",
        "All strategies route only Low case updates. Each Low update applies the mask to the averaged case gradient; this is case-update-adaptive, not per-patch sample-adaptive. The 18 groups are six decoder layers × Q/K/V, with 9 selected per Low update. High updates continue to update all groups.",
        "",
        *route_lines,
        "",
        "## Protocol and interpretation limits",
        "",
        "- E5/R0 reuses E3 raw Low baseline: same frozen High cache and raw frozen-source Bernoulli-KL Low objective; all Q/K/V groups update. E4 was excluded from R0 because it lowered mean Dice and Recall versus E3.",
        "- E6 selects the first nine sorted groups; E7 selects a seeded random nine; E8 selects the nine groups with largest Low-gradient norms; E9 selects the nine groups with largest cosine scores against a detached independent frozen-cache High reference gradient.",
        "- The High reference is one independent effective High patch per Low update, cycled over available patches. This is a compute-limited approximation to a multi-batch High-set gradient estimate; the reference is a target-domain consensus, not ground truth.",
        "- Unselected gradients are set to None before AdamW.step, preventing momentum and weight decay from moving those groups. The CSV records each selected group and its score.",
        "- The first E8/E9 pass (preserved under `stage_d_firstpass/`) exposed nonfinite group scores in 3 of 239 Low updates for each run. The safe_v2 reruns make nonfinite scores rank below finite scores; when every score is nonfinite, sorted group order is the deterministic tie-break. This changed the selected set in one Low update per method; the final Dice was unchanged to six decimals. A 40-step intermediate run was stopped before its first checkpoint when the CSV schema changed; it is preserved separately and excluded.",
        "- A single seed and 1000 updates are exploratory. E9's extra reference forward also changes the stochastic forward-call sequence; group routing results need multi-seed replication before a method claim.",
        "- Fixed/random/norm/cosine variants are compared with the same 50% active-group budget. Final adapter L2 offsets relative to the E3/R0 checkpoint are reported separately from routing frequencies; they do not equal the cumulative optimizer step length.",
        "- Per-case metrics: `per_case_metrics_stage_d.csv`; summary: `summary_stage_d.csv`; routing frequency: `routing_frequency_stage_d.csv`; adapter offsets: `parameter_offsets_stage_d.csv`.",
        "",
    ]
    (analysis / "REPORT_STAGE_D.md").write_text("\n".join(lines), encoding="utf-8")
    print(analysis / "REPORT_STAGE_D.md")


if __name__ == "__main__":
    main()
