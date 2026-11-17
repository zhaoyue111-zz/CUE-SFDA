from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research_v2"
FILES = {
    "E0": OUT / "E0_fixed" / "evaluation" / "metrics_per_case.csv",
    "E1": OUT / "E1_eval" / "evaluation" / "metrics_per_case.csv",
    "H1v1": OUT / "H1v1_eval" / "evaluation" / "metrics_per_case.csv",
}


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["case"]: {key: float(row[key]) for key in ("dice", "precision", "recall", "iou")}
            for row in csv.DictReader(handle)
        }


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, repeats: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    results = {name: read_metrics(path) for name, path in FILES.items()}
    ids = sorted(results["E0"])
    if any(set(values) != set(ids) for values in results.values()):
        raise ValueError("E0, E1 and H1v1 evaluation case IDs do not match")
    fields = ["case_id"]
    for name in results:
        fields += [f"{name}_{metric}" for metric in ("dice", "precision", "recall", "iou")]
    fields += ["E1_minus_E0_dice", "H1v1_minus_E1_dice", "H1v1_minus_E0_dice"]
    rows = []
    for case in ids:
        row = {"case_id": case}
        for name in results:
            row.update({f"{name}_{metric}": results[name][case][metric] for metric in ("dice", "precision", "recall", "iou")})
        row["E1_minus_E0_dice"] = row["E1_dice"] - row["E0_dice"]
        row["H1v1_minus_E1_dice"] = row["H1v1_dice"] - row["E1_dice"]
        row["H1v1_minus_E0_dice"] = row["H1v1_dice"] - row["E0_dice"]
        rows.append(row)
    report_dir = OUT / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "per_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for name in results:
        summary[name] = {
            metric: float(np.mean([results[name][case][metric] for case in ids]))
            for metric in ("dice", "precision", "recall", "iou")
        }
    import torch
    prototype_path = OUT / "shared_high_pseudo_cache" / "case_prototypes.pt"
    prototypes = torch.load(prototype_path, map_location="cpu", weights_only=True)
    vectors = torch.nn.functional.normalize(torch.stack(list(prototypes.values())).float(), dim=1)
    similarities = vectors @ vectors.T
    off_diagonal = similarities[~torch.eye(len(vectors), dtype=torch.bool)]
    prototype_similarity = {
        "prototype_count": len(vectors),
        "mean_pairwise_cosine": float(off_diagonal.mean()),
        "median_pairwise_cosine": float(off_diagonal.median()),
        "p10_pairwise_cosine": float(torch.quantile(off_diagonal, 0.1)),
        "p90_pairwise_cosine": float(torch.quantile(off_diagonal, 0.9)),
    }
    paired = {}
    for key, left, right in (("E1-E0", "E1", "E0"), ("H1v1-E1", "H1v1", "E1"), ("H1v1-E0", "H1v1", "E0")):
        delta = np.asarray([results[left][case]["dice"] - results[right][case]["dice"] for case in ids])
        low, high = bootstrap_mean_ci(delta)
        paired[key] = {
            "mean_dice_delta": float(delta.mean()),
            "median_dice_delta": float(np.median(delta)),
            "bootstrap_95_ci": [low, high],
            "improved_cases": int((delta > 0).sum()),
            "unchanged_cases": int((delta == 0).sum()),
            "declined_cases": int((delta < 0).sum()),
            "delta_gt_0_02": int((delta > 0.02).sum()),
            "delta_lt_minus_0_02": int((delta < -0.02).sum()),
        }
    text = [
        "# Research v2 Stage A/B results",
        "",
        "The 44-case held-out split was reused for earlier method and checkpoint selection. These fixed-iteration scores are exploratory labeled development results, not an untouched final test. No checkpoint was selected using these metrics.",
        "",
        "## Aggregate metrics",
        "",
        "| Run | Dice | Precision | Recall | IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summary.items():
        text.append(f"| {name} | {metrics['dice']:.6f} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['iou']:.6f} |")
    text += ["", "## Paired Dice changes", "", "| Contrast | Mean Δ | Median Δ | Bootstrap 95% CI | Improved | Unchanged | Declined | Δ>0.02 | Δ<−0.02 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, values in paired.items():
        lo, hi = values["bootstrap_95_ci"]
        text.append(f"| {key} | {values['mean_dice_delta']:.6f} | {values['median_dice_delta']:.6f} | [{lo:.6f}, {hi:.6f}] | {values['improved_cases']} | {values['unchanged_cases']} | {values['declined_cases']} | {values['delta_gt_0_02']} | {values['delta_lt_minus_0_02']} |")
    text += ["", "## Protocol notes", "", "- E0: frozen VoxTell; E1: fixed 500-step high-only self-training; H1-v1: same initialization, cases, pseudo-label cache and budget, plus cross-case top-5 relation loss (λ=0.1, temperature=0.1).", "- Prompt: `liver cancer`; only decoder Q/K/V LoRA parameters were trainable in E1/H1-v1.", "- Evaluation used the repository's fixed sliding-window inference, 192³ patches, 0.5 tile step and Gaussian blending.", "- The 44 cases are never used for training or checkpoint selection in this run. Metrics are post-hoc diagnostics.", f"- Frozen teacher produced {prototype_similarity['prototype_count']} usable foreground case prototypes from 59 High cases; pairwise cosine mean/median was {prototype_similarity['mean_pairwise_cosine']:.4f}/{prototype_similarity['median_pairwise_cosine']:.4f} (p10–p90 {prototype_similarity['p10_pairwise_cosine']:.4f}–{prototype_similarity['p90_pairwise_cosine']:.4f}), indicating substantial shared feature direction and a potential weak-discrimination limitation for H1-v1.", f"- Engineering target check (mean Dice ≥0.38): E0={summary['E0']['dice'] >= 0.38}, E1={summary['E1']['dice'] >= 0.38}, H1-v1={summary['H1v1']['dice'] >= 0.38}.", "- Per-case paired metrics: `per_case_metrics.csv`.", ""]
    (report_dir / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Wrote {report_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
