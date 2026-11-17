from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research_v2"


def read(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["case"]: {k: float(r[k]) for k in ("dice", "precision", "recall", "iou")} for r in csv.DictReader(handle)}


def main() -> None:
    e3 = read(OUT / "E3_eval" / "evaluation" / "metrics_per_case.csv")
    e4 = read(OUT / "E4_eval" / "evaluation" / "metrics_per_case.csv")
    cases = sorted(e3)
    if set(e3) != set(e4):
        raise ValueError("E3 and E4 evaluation cases differ")
    delta = np.asarray([e4[c]["dice"] - e3[c]["dice"] for c in cases])
    rng = np.random.default_rng(42)
    sampled = rng.integers(0, len(cases), size=(10000, len(cases)))
    boot = delta[sampled].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    means = {
        name: {metric: float(np.mean([rows[c][metric] for c in cases])) for metric in ("dice", "precision", "recall", "iou")}
        for name, rows in (("E3", e3), ("E4", e4))
    }
    out = OUT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_case_metrics_stage_c.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "E3_dice", "E4_dice", "delta_dice", "E3_precision", "E4_precision", "E3_recall", "E4_recall"])
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case,
                "E3_dice": e3[case]["dice"],
                "E4_dice": e4[case]["dice"],
                "delta_dice": e4[case]["dice"] - e3[case]["dice"],
                "E3_precision": e3[case]["precision"],
                "E4_precision": e4[case]["precision"],
                "E3_recall": e3[case]["recall"],
                "E4_recall": e4[case]["recall"],
            })
    lines = [
        "# Stage C: frozen High memory transfer to Low",
        "",
        "The 44-case held-out set has been reused in prior exploration; these are fixed-iteration labeled development diagnostics, not an untouched final test. GT was not used in training or checkpoint selection.",
        "",
        "| Run | Dice | Precision | Recall | IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in means.items():
        lines.append(f"| {name} | {m['dice']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['iou']:.6f} |")
    lines += [
        "",
        "## Paired E4 − E3 Dice",
        "",
        f"- Mean Δ: {delta.mean():.6f}; median Δ: {np.median(delta):.6f}.",
        f"- 95% paired bootstrap CI: [{lo:.6f}, {hi:.6f}].",
        f"- Improved / unchanged / declined: {(delta > 0).sum()} / {(delta == 0).sum()} / {(delta < 0).sum()}.",
        f"- Δ>0.02 / Δ<−0.02: {(delta > 0.02).sum()} / {(delta < -0.02).sum()}.",
        "",
        "## Protocol notes",
        "",
        "- C_joint fixed train split: 101 train cases (59 High, 42 Low); 44 held-out cases.",
        "- Both runs use the same frozen-source High pseudo-label cache, 1000 updates, seed, Q/K/V LoRA set, and raw-image frozen-source Bernoulli-KL Low baseline.",
        "- E4 adds only the foreground-masked D5 cosine relation to frozen High case prototypes (weight 0.1). If a Low frozen-source patch has no foreground, the relation term is skipped; no GT or response-based gate is used.",
        "- The repository's default low branch uses augmented-view consistency. For this plan's raw-image-only requirement, E3/E4 used the explicit opt-in raw frozen-source KL path; default behavior was not changed.",
        "- Evaluation used 192³ sliding windows, tile step 0.5, and Gaussian blending.",
        "- Per-case paired results: `per_case_metrics_stage_c.csv`.",
        "",
    ]
    (out / "REPORT_STAGE_C.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "REPORT_STAGE_C.md")


if __name__ == "__main__":
    main()
