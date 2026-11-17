from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research_v2"
FILES = {
    "E0": OUT / "E0_fixed" / "evaluation" / "metrics_per_case.csv",
    "E1_500": OUT / "E1_eval" / "evaluation" / "metrics_per_case.csv",
    "H1v1_500": OUT / "H1v1_eval" / "evaluation" / "metrics_per_case.csv",
    "E1_1000": OUT / "E1_eval_1000" / "evaluation" / "metrics_per_case.csv",
    "H1v1_1000": OUT / "H1v1_eval_1000" / "evaluation" / "metrics_per_case.csv",
}


def load(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["case"]: {k: float(r[k]) for k in ("dice", "precision", "recall", "iou")} for r in csv.DictReader(handle)}


def ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    picks = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[picks].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main() -> None:
    metrics = {name: load(path) for name, path in FILES.items()}
    cases = sorted(metrics["E0"])
    if any(set(v) != set(cases) for v in metrics.values()):
        raise ValueError("Evaluation case IDs differ across runs")
    report_dir = OUT / "analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = ["case_id"]
    for name in metrics:
        fields += [f"{name}_{m}" for m in ("dice", "precision", "recall", "iou")]
    for contrast in ("E1_500_minus_E0", "H1v1_500_minus_E1_500", "E1_1000_minus_E0", "H1v1_1000_minus_E1_1000", "E1_1000_minus_E1_500"):
        fields.append(contrast)
    rows = []
    for case in cases:
        row = {"case_id": case}
        for name in metrics:
            row.update({f"{name}_{m}": metrics[name][case][m] for m in ("dice", "precision", "recall", "iou")})
        for contrast, a, b in (
            ("E1_500_minus_E0", "E1_500", "E0"),
            ("H1v1_500_minus_E1_500", "H1v1_500", "E1_500"),
            ("E1_1000_minus_E0", "E1_1000", "E0"),
            ("H1v1_1000_minus_E1_1000", "H1v1_1000", "E1_1000"),
            ("E1_1000_minus_E1_500", "E1_1000", "E1_500"),
        ):
            row[contrast] = row[f"{a}_dice"] - row[f"{b}_dice"]
        rows.append(row)
    with (report_dir / "per_case_metrics_500_1000.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    means = {name: {m: float(np.mean([metrics[name][case][m] for case in cases])) for m in ("dice", "precision", "recall", "iou")} for name in metrics}
    contrasts = []
    for label, a, b in (
        ("E1@500 - E0", "E1_500", "E0"),
        ("H1-v1@500 - E1@500", "H1v1_500", "E1_500"),
        ("E1@1000 - E0", "E1_1000", "E0"),
        ("H1-v1@1000 - E1@1000", "H1v1_1000", "E1_1000"),
        ("E1@1000 - E1@500", "E1_1000", "E1_500"),
    ):
        d = np.asarray([metrics[a][c]["dice"] - metrics[b][c]["dice"] for c in cases])
        lo, hi = ci(d)
        contrasts.append((label, float(d.mean()), float(np.median(d)), lo, hi, int((d > 0).sum()), int((d < 0).sum()), int((d > .02).sum()), int((d < -.02).sum())))
    text = [
        "# Research v2 Stage A/B: fixed 500 and 1000 step results",
        "",
        "All 44 evaluation cases were previously used for method exploration and checkpoint selection. In this run the 500- and 1000-step checkpoints were fixed by iteration, with no GT-based selection; these are exploratory labeled development results, not an untouched final test.",
        "",
        "## Aggregate metrics",
        "",
        "| Run | Dice | Precision | Recall | IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, x in means.items():
        text.append(f"| {name} | {x['dice']:.6f} | {x['precision']:.6f} | {x['recall']:.6f} | {x['iou']:.6f} |")
    text += ["", "## Paired Dice deltas", "", "| Contrast | Mean | Median | 95% bootstrap CI | Improved | Declined | Δ>0.02 | Δ<−0.02 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, mean, median, lo, hi, inc, dec, big_inc, big_dec in contrasts:
        text.append(f"| {label} | {mean:.6f} | {median:.6f} | [{lo:.6f}, {hi:.6f}] | {inc} | {dec} | {big_inc} | {big_dec} |")
    text += ["", "## Interpretation", "", f"- Engineering target mean Dice ≥0.38: E1@1000={means['E1_1000']['dice'] >= .38}; H1-v1@1000={means['H1v1_1000']['dice'] >= .38}.", "- Q/K/V LoRA only; fixed `liver cancer` prompt; E1 and H1-v1 share frozen-source pseudo labels and 1000 updates. H1-v1 adds only the cross-case top-5 relation loss.", "- GT is post-hoc only. No low branch, Stage 2, SAT, routing update, or dynamic QKV selection was used in these experiments.", "- Evaluation uses the repository's 192³ sliding-window inference, 0.5 tile step and Gaussian blending.", "- Per-case deltas are in `per_case_metrics_500_1000.csv`.", ""]
    (report_dir / "REPORT_500_1000.md").write_text("\n".join(text), encoding="utf-8")
    print(report_dir / "REPORT_500_1000.md")


if __name__ == "__main__":
    main()
