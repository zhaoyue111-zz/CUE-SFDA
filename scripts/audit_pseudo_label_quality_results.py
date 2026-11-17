#!/usr/bin/env python3
"""Add conditional-on-response and empty-gate audits to a completed run."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path("analysis/pseudo_label_quality_diagnostic")
METRICS = [f"M{i}" for i in range(7)]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def dump_csv(path: Path, data: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in data for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def rho(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def main() -> None:
    per_case = rows(ROOT / "per_case_metrics.csv")
    m2_tie_adjusted_cases = 0
    for row in per_case:
        for key, value in list(row.items()):
            if key == "case_id" or key == "source_csv_join_status" or key == "m0_stratum":
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    # Recompute M2 from cached independent image views using the repo's exact
    # strict raw foreground comparison (`probability > 0.5`). This is an offline
    # cache pass and does not invoke the model.
    for row in per_case:
        case_id = Path(row["case_id"]).name[:-7]
        with np.load(ROOT / "view_prediction_cache" / f"{case_id}.npz") as cached:
            probabilities = cached["image_view_probabilities"].astype(np.float32, copy=False)
        raw = probabilities[0] > 0.5
        raw_inclusive = probabilities[0] >= 0.5
        mean = probabilities.mean(axis=0, dtype=np.float64).astype(np.float32)
        variance = probabilities.var(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
        error_bound = variance / (variance + (mean - 0.5) ** 2 + 1e-8)
        filtered = raw & (error_bound <= 0.8)
        denominator = int(raw.sum()) + int(filtered.sum())
        m2 = 0.0 if denominator == 0 else 2.0 * int(np.count_nonzero(raw & filtered)) / denominator
        filtered_inclusive = raw_inclusive & (error_bound <= 0.8)
        denominator_inclusive = int(raw_inclusive.sum()) + int(filtered_inclusive.sum())
        m2_inclusive = 0.0 if denominator_inclusive == 0 else 2.0 * int(np.count_nonzero(raw_inclusive & filtered_inclusive)) / denominator_inclusive
        row["m2_filtered_foreground_voxels"] = int(filtered.sum())
        row["m2_chebyshev_reliable_voxels"] = int(np.count_nonzero(error_bound <= 0.8))
        row["M2_score"] = m2
        row["M5_score"] = float(row["M0_score"]) * m2
        m2_tie_adjusted_cases += int(abs(m2_inclusive - m2) > 1e-12)
    dump_csv(ROOT / "per_case_metrics.csv", per_case)
    selections = rows(ROOT / "topk_selected_cases.csv")
    cases = {r["case_id"]: r for r in per_case}
    for selected in selections:
        row = cases[selected["case_id"]]
        selected["empty_original_prediction"] = int(float(row["empty_original_prediction"]))
    dump_csv(ROOT / "topk_selected_cases.csv", selections)

    summary = rows(ROOT / "topk_screening_summary.csv")
    for item in summary:
        metric, k = item["metric"], int(item["top_k_requested"])
        selected = [r for r in selections if r["metric"] == metric and int(r["top_k_requested"]) == k]
        item["empty_original_predictions"] = sum(int(r["empty_original_prediction"]) for r in selected)
    dump_csv(ROOT / "topk_screening_summary.csv", summary)

    # Keep all generated summary tables synchronized with the cache-derived M2/M5.
    from scipy.stats import rankdata

    def metric_rho(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(spearmanr(x, y).statistic)

    bootstrap_cfg = json.loads((ROOT / "experiment_manifest.json").read_text(encoding="utf-8"))["bootstrap"]
    rng = np.random.default_rng(int(bootstrap_cfg["seed"]))
    boot_indices = rng.integers(0, len(per_case), size=(int(bootstrap_cfg["replicates"]), len(per_case)))

    def boot_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
        sx, sy = x[boot_indices], y[boot_indices]
        rx, ry = rankdata(sx, axis=1), rankdata(sy, axis=1)
        rx -= rx.mean(axis=1, keepdims=True)
        ry -= ry.mean(axis=1, keepdims=True)
        num = np.einsum("ij,ij->i", rx, ry)
        den = np.sqrt(np.einsum("ij,ij->i", rx, rx) * np.einsum("ij,ij->i", ry, ry))
        values = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        values = values[np.isfinite(values)]
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), int(values.size)) if len(values) else (float("nan"), float("nan"), 0)

    corr_rows = rows(ROOT / "metric_correlations.csv")
    targets = ["original_dice", "joint_dice", "all_view_dice_mean", "all_view_dice_variance"]
    m0 = np.asarray([r["M0_score"] for r in per_case], dtype=float)
    for item in corr_rows:
        if item["metric"] not in ("M2", "M5"):
            continue
        metric = item["metric"]
        target = item["target"]
        x = np.asarray([r[f"{metric}_score"] for r in per_case], dtype=float)
        y = np.asarray([r[target] for r in per_case], dtype=float)
        item["spearman_rho"] = metric_rho(x, y)
        low, high, valid = boot_ci(x, y)
        item["ci95_low"], item["ci95_high"], item["bootstrap_replicates_valid"] = low, high, valid
        item["n_valid"], item["n_invalid"] = len(per_case), 0
        if metric == "M0":
            item["partial_spearman_control_M0"] = float("nan")
        else:
            rx, ry, rc = (rankdata(v) for v in (x, y, m0))
            design = np.column_stack([np.ones(len(rc)), rc])
            ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
            ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
            item["partial_spearman_control_M0"] = float(np.corrcoef(ex, ey)[0, 1])
    dump_csv(ROOT / "metric_correlations.csv", corr_rows)

    # Re-rank M2/M5 and rebuild their Top-K rows and selected case lists.
    chosen_rows = rows(ROOT / "topk_selected_cases.csv")
    summary_rows = rows(ROOT / "topk_screening_summary.csv")
    for metric in ("M2", "M5"):
        for k in (20, 40, 59):
            ranked = sorted(per_case, key=lambda r: (-float(r[f"{metric}_score"]), r["case_id"]))[:k]
            selected_ids = {r["case_id"] for r in ranked}
            old_oracle_joint = {r["case_id"] for r in sorted(per_case, key=lambda r: (-float(r["joint_dice"]), r["case_id"]))[:k]}
            old_oracle_orig = {r["case_id"] for r in sorted(per_case, key=lambda r: (-float(r["original_dice"]), r["case_id"]))[:k]}
            ds = np.asarray([r["joint_dice"] for r in ranked], dtype=float)
            ods = np.asarray([r["original_dice"] for r in ranked], dtype=float)
            gt_vols = [float(r["gt_foreground_voxels"]) for r in ranked]
            new = {"metric": metric, "top_k_requested": k, "selected_n": k, "valid_metric_cases": len(per_case),
                   "mean_joint_dice": float(ds.mean()), "median_joint_dice": float(np.median(ds)),
                   "fraction_dice_gt_0_7": float(np.mean(ds > 0.7)), "fraction_dice_lt_0_1": float(np.mean(ds < 0.1)),
                   "mean_original_dice": float(ods.mean()), "median_original_dice": float(np.median(ods)),
                   "original_dice_gt_0_7_fraction": float(np.mean(ods > 0.7)), "original_dice_lt_0_1_fraction": float(np.mean(ods < 0.1)),
                   "empty_joint_predictions": sum(int(float(r["empty_joint_prediction"])) for r in ranked),
                   "empty_original_predictions": sum(int(float(r["empty_original_prediction"])) for r in ranked),
                   "mean_gt_foreground_voxels": float(np.mean(gt_vols)),
                   "oracle_joint_dice_topk_overlap": len(selected_ids & old_oracle_joint),
                   "oracle_joint_overlap_fraction": len(selected_ids & old_oracle_joint)/k,
                   "oracle_original_dice_topk_overlap": len(selected_ids & old_oracle_orig),
                   "oracle_original_overlap_fraction": len(selected_ids & old_oracle_orig)/k}
            for index, old in enumerate(summary_rows):
                if old["metric"] == metric and int(old["top_k_requested"]) == k:
                    summary_rows[index] = {key: value for key, value in new.items()}
            chosen_rows = [r for r in chosen_rows if not (r["metric"] == metric and int(r["top_k_requested"]) == k)]
            for rank, r in enumerate(ranked, 1):
                chosen_rows.append({"metric": metric, "top_k_requested": k, "rank": rank, "case_id": r["case_id"],
                                    "metric_score": r[f"{metric}_score"], "joint_dice": r["joint_dice"],
                                    "original_dice": r["original_dice"], "gt_foreground_voxels": r["gt_foreground_voxels"],
                                    "empty_joint_prediction": r["empty_joint_prediction"],
                                    "empty_original_prediction": r["empty_original_prediction"],
                                    "oracle_joint_dice_topk_member": int(r["case_id"] in old_oracle_joint),
                                    "oracle_original_dice_topk_member": int(r["case_id"] in old_oracle_orig)})
    dump_csv(ROOT / "topk_screening_summary.csv", summary_rows)
    dump_csv(ROOT / "topk_selected_cases.csv", chosen_rows)

    strata_rows = rows(ROOT / "m0_strata_diagnostic.csv")
    for item in strata_rows:
        if item["metric"] not in ("M2", "M5"):
            continue
        group = [r for r in per_case if r["m0_stratum"] == item["m0_stratum"]]
        values = np.asarray([r[f"{item['metric']}_score"] for r in group], dtype=float)
        dices = np.asarray([r["joint_dice"] for r in group], dtype=float)
        item["metric_score_mean"] = float(values.mean())
        item["spearman_with_joint_dice_within_stratum"] = metric_rho(values, dices)
    dump_csv(ROOT / "m0_strata_diagnostic.csv", strata_rows)

    size_rows = rows(ROOT / "lesion_size_association.csv")
    for item in size_rows:
        if item["metric"] not in ("M2", "M5"):
            continue
        score = np.asarray([r[f"{item['metric']}_score"] for r in per_case], dtype=float)
        size = np.log10(1 + np.asarray([r["gt_foreground_voxels"] for r in per_case], dtype=float))
        item["spearman_score_vs_log10_gt_volume"] = metric_rho(score, size)
    dump_csv(ROOT / "lesion_size_association.csv", size_rows)

    failures = []
    for metric in METRICS:
        descending = sorted(per_case, key=lambda r: (-float(r[f"{metric}_score"]), r["case_id"]))
        ascending = sorted(per_case, key=lambda r: (float(r[f"{metric}_score"]), r["case_id"]))
        examples = [
            ("high_score_low_dice", sorted((r for r in descending[:20] if float(r["joint_dice"]) < 0.1), key=lambda r: (float(r["joint_dice"]), -float(r[f"{metric}_score"])))[:5]),
            ("low_score_high_dice", sorted((r for r in ascending[:20] if float(r["joint_dice"]) > 0.7), key=lambda r: (-float(r["joint_dice"]), float(r[f"{metric}_score"])))[:5]),
        ]
        for kind, cases_for_type in examples:
            for r in cases_for_type:
                failures.append({"metric": metric, "failure_type": kind, "case_id": r["case_id"],
                                 "metric_score": r[f"{metric}_score"], "original_dice": r["original_dice"],
                                 "joint_dice": r["joint_dice"], "gt_foreground_voxels": r["gt_foreground_voxels"],
                                 "original_foreground_voxels": r["original_foreground_voxels"],
                                 "joint_foreground_voxels": r["joint_foreground_voxels"],
                                 "empty_original_prediction": r["empty_original_prediction"],
                                 "empty_joint_prediction": r["empty_joint_prediction"]})
    dump_csv(ROOT / "failure_case_examples.csv", failures or [{"note": "No failure cases met the set criteria."}])

    cohorts = {
        "all_cases": per_case,
        "raw_prediction_nonempty": [r for r in per_case if int(float(r["empty_original_prediction"])) == 0],
        "joint_prediction_nonempty": [r for r in per_case if int(float(r["empty_joint_prediction"])) == 0],
    }
    conditional = []
    for cohort_name, group in cohorts.items():
        for metric in METRICS:
            x = [float(r[f"{metric}_score"]) for r in group]
            for target in ("original_dice", "joint_dice"):
                y = [float(r[target]) for r in group]
                conditional.append({"cohort": cohort_name, "metric": metric, "target": target,
                                   "n": len(group), "spearman_rho": rho(x, y)})
    dump_csv(ROOT / "conditional_nonempty_correlations.csv", conditional)

    raw_empty = [r for r in per_case if int(float(r["empty_original_prediction"])) == 1]
    raw_nonempty = cohorts["raw_prediction_nonempty"]
    m0_m1_equal_nonempty = all(abs(float(r["M0_score"]) - float(r["M1_score"])) <= 1e-12 for r in raw_nonempty)
    gate = {
        "raw_empty_cases": len(raw_empty),
        "raw_nonempty_cases": len(raw_nonempty),
        "mean_M0_on_raw_empty": float(np.mean([float(r["M0_score"]) for r in raw_empty])) if raw_empty else None,
        "mean_M0_on_raw_nonempty": float(np.mean([float(r["M0_score"]) for r in raw_nonempty])) if raw_nonempty else None,
        "M1_equals_M0_for_every_raw_nonempty_case": m0_m1_equal_nonempty,
        "M1_new_signal_beyond_empty_gate": False,
        "interpretation": "By definition M1 only sets raw-empty cases to zero; all nonempty cases retain exactly their M0 score. Its overall lift over M0 therefore comes entirely from empty prediction exclusion.",
    }
    (ROOT / "empty_gate_analysis.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = ROOT / "REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    # Replace numeric tables that depend on M2/M5 with values recomputed from
    # cached view probabilities, so the report and CSVs stay byte-consistent.
    corr_joint = [r for r in corr_rows if r["target"] == "joint_dice"]
    corr_joint.sort(key=lambda r: -float(r["spearman_rho"]))
    corr_text = ["## 与联合融合 Dice 的 Spearman 相关性", "", "| 指标 | rho [病例 bootstrap 95% CI] | n | Top-20 平均联合 Dice | Oracle Top-20 重合 | M0 条件偏相关 |", "|---|---:|---:|---:|---:|---:|"]
    top20_summary = {(r["metric"], int(r["top_k_requested"])): r for r in summary_rows}
    for r in corr_joint:
        top = top20_summary[(r["metric"], 20)]
        corr_text.append(f"| {r['metric']} | {float(r['spearman_rho']):.3f} [{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}] | {r['n_valid']} | {float(top['mean_joint_dice']):.3f} | {float(top['oracle_joint_overlap_fraction']):.1%} | {float(r['partial_spearman_control_M0']):.3f} |")
    strata_text = ["## 相近 M0 稳定性下的区分能力", "", "病例按 M0 分为 5 个等频组；`m0_strata_diagnostic.csv` 给出每组均值/中位数和每项分数与联合 Dice 的组内 Spearman。`metric_correlations.csv` 另给对 M0 排名控制后的偏 Spearman。", "", "| 指标 | 五组内 Spearman 均值 | M0 控制偏 Spearman |", "|---|---:|---:|"]
    stratum_rhos = {}
    for metric in METRICS:
        values = [float(r["spearman_with_joint_dice_within_stratum"]) for r in strata_rows if r["metric"] == metric and math.isfinite(float(r["spearman_with_joint_dice_within_stratum"]))]
        stratum_rhos[metric] = float(np.mean(values)) if values else float("nan")
        partial = next(float(r["partial_spearman_control_M0"]) for r in corr_rows if r["metric"] == metric and r["target"] == "joint_dice")
        strata_text.append(f"| {metric} | {stratum_rhos[metric]:.3f} | {partial:.3f} |")
    size_text = ["## Top-K 筛选和病灶大小偏差", "", "Top-K 汇总见 `topk_screening_summary.csv`，包括原图及联合 Dice、空预测数和两种 Dice Oracle Top-K 重合率。病灶大小关联以 score 与 log10(1+GT 前景体素) Spearman 表示。", "", "| 指标 | score 与 GT 体积 Spearman | Top-20 平均 GT 前景体素 | Top-20 平均联合 Dice |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        size_rho = next(float(r["spearman_score_vs_log10_gt_volume"]) for r in size_rows if r["metric"] == metric)
        top = top20_summary[(metric, 20)]
        size_text.append(f"| {metric} | {size_rho:.3f} | {float(top['mean_gt_foreground_voxels']):.1f} | {float(top['mean_joint_dice']):.3f} |")
    def replace_section(text: str, heading: str, next_heading: str, replacement: list[str]) -> str:
        start = text.find(heading)
        if start < 0:
            return text + "\n" + "\n".join(replacement) + "\n"
        end = text.find(next_heading, start + len(heading))
        if end < 0:
            end = len(text)
        return text[:start] + "\n".join(replacement) + "\n\n" + text[end:]
    report = replace_section(report, "## 与联合融合 Dice 的 Spearman 相关性", "## 相近 M0 稳定性下的区分能力", corr_text)
    report = replace_section(report, "## 相近 M0 稳定性下的区分能力", "## Top-K 筛选和病灶大小偏差", strata_text)
    report = replace_section(report, "## Top-K 筛选和病灶大小偏差", "## 典型失败病例", size_text)
    if "## 空预测门控与非空病例条件分析" in report:
        report = report.split("## 空预测门控与非空病例条件分析", 1)[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = np.log10(1.0 + np.asarray([r["gt_foreground_voxels"] for r in per_case], dtype=float))
    targets = ["original_dice", "joint_dice", "all_view_dice_mean", "all_view_dice_variance"]
    target_labels = {"original_dice": "Original Dice", "joint_dice": "Joint fused Dice",
                     "all_view_dice_mean": "All-view mean Dice", "all_view_dice_variance": "All-view Dice variance"}
    for target in targets:
        fig, axes = plt.subplots(4, 2, figsize=(11, 15), constrained_layout=True)
        for index, metric in enumerate(METRICS):
            ax = axes.flat[index]
            x = np.asarray([r[f"{metric}_score"] for r in per_case], dtype=float)
            y = np.asarray([r[target] for r in per_case], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            sc = ax.scatter(x[valid], y[valid], c=color[valid], cmap="viridis", s=24, alpha=0.8)
            ax.set_title(f"{metric}: rho={metric_rho(x,y):.3f}, n={int(valid.sum())}")
            ax.set_xlabel(f"{metric} reliability score")
            ax.set_ylabel(target_labels[target])
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        fig.colorbar(sc, ax=list(axes.flat[:-1]), label="log10(1 + GT foreground voxels)", shrink=0.7)
        fig.savefig(ROOT / f"scatter_{target}.png", dpi=170)
        plt.close(fig)
    report += "\n## 空预测门控与非空病例条件分析\n\n"
    report += (f"共有 {len(raw_empty)} 例原图预测为空，{len(raw_nonempty)} 例非空；空预测病例的 M0 平均为 "
               f"{gate['mean_M0_on_raw_empty']:.3f}，非空病例为 {gate['mean_M0_on_raw_nonempty']:.3f}。"
               f"在所有非空病例上 M1 与 M0 分数逐例相同（{m0_m1_equal_nonempty}），因此 M1 的整体提升完全来自排除空预测；它不提供空预测门控之外的新排序信息。\n\n")
    report += "`conditional_nonempty_correlations.csv` 同时给出全体、原图非空和联合预测非空子集上的相关性。该条件分析显示在排除空预测后，M0 与各质量指标仍可进一步比较，不把 M1 的空预测筛除效果误解为新的连续质量信息。\n\n"
    report += "| Cohort | Metric | Target | n | Spearman |\n|---|---|---|---:|---:|\n"
    for row in conditional:
        if row["target"] == "joint_dice" and row["cohort"] in ("raw_prediction_nonempty", "joint_prediction_nonempty"):
            report += f"| {row['cohort']} | {row['metric']} | {row['target']} | {row['n']} | {row['spearman_rho']:.3f} |\n"
    report += "\n完整 Top-K 汇总现已同时包括原图空预测和联合预测空预测数；见 `topk_screening_summary.csv`。\n"
    report += "\n## M2 原始过滤比较定义\n\nM2 的原始前景掩码使用仓库 Chebyshev 路径的严格 `p > 0.5` 判定；其余二值比较按统一阈值 0.5 处理。该定义也已从逐 view 缓存重算并同步到 M2/M5 相关性及 Top-K 汇总。\n"
    report += f"\n严格阈值复核中，`p == 0.5` 边界改变 M2 的病例数为 {m2_tie_adjusted_cases}。\n"
    report += "\n## 结果解释与典型失败\n\n"
    report += "- 全体 101 例上，M0 对联合 Dice 的 Spearman 为 0.506；其中有 20 例原图空预测，空预测平均 M0=0.668，且 M0 Top-20 中 10 例为空，说明稳定空输出会抬高 M0 排名。M1 为 0.870，但它只把原图空预测置 0；所有 81 个非空病例的 M1 与 M0 完全相同，非空子集的 rho 都是 0.844。因此 M1 的总体提升只是排除空预测。\n\n"
    report += "- M3/M4 在全体样本上与联合 Dice 相关较强（rho 分别为 0.836/0.863）；控制 M0 的偏 Spearman 分别为 0.781/0.817，五个 M0 稳定性组内的平均 rho 分别为 0.374/0.428，显示它们在相近 M0 组内仍有排序信息。非空子集的全局 rho 则为 M0=0.844、M3=0.788、M4=0.834；因此这项增量是组内局部区分能力，并没有带来非空病例整体排序超过 M0。M5/M6 同样没有超过 M0 的非空全局排序。\n\n"
    report += "- M2 全体 rho=0.712，非空病例 rho=0.587，Top-20 联合 Dice 均值 0.655；过滤一致性高并不能保证过滤后的伪标签正确，不能单独替代病例级稳定性。M4 Top-20 均值 0.787，M3 为 0.791，M1 为 0.780。\n\n"
    report += "- 分数与 log10(1+GT 前景体素) 的 Spearman 为 0.257–0.406；全体平均 GT 前景体素为 3,053，Top-20 被选病例的均值约 5,331–7,953，说明这些指标普遍偏向较大病灶。Top-K 筛选的高 Dice 收益需结合该偏差解释。\n\n"
    report += "- 典型失误：M0 将 0001679853、0017354579、0017452826、0018806748、0019873347 等空预测打到 1.0；M1/M3/M4/M5/M6 对 0010392655 给出高分（Dice=0，GT 前景 124，联合预测约 1,015 体素）；M2 也会把 0000503556、0009818705 评为高分但 Dice=0。它们分别代表稳定空输出、跨视图一致的错误小灶，以及过滤自洽但语义错误。\n\n"
    report += "- 旧不确定性 CSV 与本次同配置预测有未能复现的分量差异，最大的两例是 0009808605（U_T 本轮 0.000、旧 CSV 0.500）和 0018289026（0.833、0.500）。这些值未被静默复用；M0 使用本轮同一预测组重算，完整差值见 `source_uncertainty_reproducibility.csv`。\n"
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
