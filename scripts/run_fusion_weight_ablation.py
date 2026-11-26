#!/usr/bin/env python3
"""Offline alpha sweep using existing image-level per-case scores only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

METRICS = ["I1_Pairwise_Hard", "I2_Pairwise_Soft", "I3_VVC"]
ALPHAS = np.round(np.arange(0, 1.0001, .1), 1)
TOPKS = [10, 20, 30, 40, 49, 59]
SEED = 20261126


def corr(fn, x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    r = fn(x, y)
    return float(r.statistic if hasattr(r, "statistic") else r[0])


def aurc(dice, confidence):
    order = np.lexsort((np.arange(len(dice)), -confidence))
    risks = 1.0 - dice[order]
    curve = np.cumsum(risks) / np.arange(1, len(risks) + 1)
    coverage = np.arange(1, len(risks) + 1, dtype=float) / len(risks)
    return float(np.trapezoid(np.r_[curve[0], curve], np.r_[0.0, coverage]))


def get_auc(labels, confidence):
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(labels, confidence)), float(average_precision_score(labels, confidence))


def rank_metrics(dice, conf):
    dice = np.asarray(dice, dtype=float)
    conf = np.asarray(conf, dtype=float)
    out = {
        "spearman": corr(spearmanr, dice, conf),
        "pearson": corr(pearsonr, dice, conf),
        "aurc": aurc(dice, conf),
    }
    for name, labels in (("dice_lt_0_1", dice < .1), ("dice_gt_0_5", dice > .5)):
        out[f"{name}_auroc"], out[f"{name}_auprc"] = get_auc(labels.astype(int), conf)
    order = np.lexsort((np.arange(len(dice)), -conf))
    for k in TOPKS:
        out[f"top{k}_mean_dice"] = float(np.mean(dice[order[:k]]))
    out["top59_mean_dice"] = out["top59_mean_dice"]
    high, low = dice[order[:59]], dice[order[59:]]
    out["high_mean_dice"] = float(high.mean())
    out["low_mean_dice"] = float(low.mean())
    return out, order


def statistic_value(dice, conf, statistic):
    if statistic == "aurc": return aurc(dice, conf)
    if statistic == "top59_mean_dice":
        order = np.lexsort((np.arange(len(dice)), -conf))
        return float(np.mean(np.asarray(dice)[order[:59]]))
    if statistic == "spearman": return corr(spearmanr, dice, conf)
    if statistic == "pearson": return corr(pearsonr, dice, conf)
    raise ValueError(statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="analysis/hierarchical_uncertainty/stage1_image_level/per_case_scores.csv")
    ap.add_argument("--out", default="analysis/hierarchical_uncertainty/stage1_image_level/fusion_weight_ablation")
    args = ap.parse_args()
    outdir = Path(args.out); figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(args.input)
    metric_rows = src[(src.metric.isin(METRICS)) & (src.aggregation == "case")]
    ids = sorted(metric_rows.case_id.unique())
    assert len(ids) == 101, f"expected 101 cases, got {len(ids)}"
    dice = (metric_rows.drop_duplicates("case_id").set_index("case_id").loc[ids].dice.to_numpy(float))
    assert (dice == 0).sum() == 42, "Expected the existing per-case table's 42 Dice-zero cases"

    # Empty prediction status is inherited from the original gated scores. There are 20 empty cases.
    base_i1 = metric_rows[metric_rows.metric == METRICS[0]]
    empty_ids = set(base_i1[(base_i1.perturbation == "Image-only") & (base_i1.gating == "gated") & (base_i1.confidence == -1)].case_id)
    assert len(empty_ids) == 20, f"expected 20 empty predictions, got {len(empty_ids)}"
    nonempty = np.array([cid not in empty_ids for cid in ids])
    assert int(nonempty.sum()) == 81
    zero_nonempty_ids = {cid for cid, d, ne in zip(ids, dice, nonempty) if d == 0 and ne}
    assert len(zero_nonempty_ids) == 22

    case_rows, summary_rows, bin_rows = [], [], []
    previous = {}
    for metric in METRICS:
        sub = metric_rows[metric_rows.metric == metric]
        maps = {}
        for perturb in ["Image-only", "Prompt-only", "Image+Prompt"]:
            part = sub[(sub.perturbation == perturb) & (sub.gating == "raw")].set_index("case_id").loc[ids]
            maps[perturb] = part.uncertainty.to_numpy(float)
        for alpha in ALPHAS:
            u = (1-alpha) * maps["Image-only"] + alpha * maps["Prompt-only"]
            conf = (1-u) if metric != "I3_VVC" else 1/(1+u)
            # Verify stored alpha=.5 and endpoints against the previous fused and pure-view rows.
            label = "Image-only" if alpha == 0 else ("Prompt-only" if alpha == 1 else ("Image+Prompt" if alpha == .5 else None))
            if label:
                existing = sub[(sub.perturbation == label) & (sub.gating == "raw")].set_index("case_id").loc[ids]
                raw_delta = float(np.max(np.abs(conf - existing.confidence.to_numpy(float))))
                assert raw_delta < 1e-10, f"Raw anchor mismatch: {metric} alpha={alpha}, delta={raw_delta}"
                previous[(metric, float(alpha))] = raw_delta
            for gating in ["raw", "gated"]:
                c = conf.copy()
                if gating == "gated": c[np.array([cid in empty_ids for cid in ids])] = 0.0
                all_scores, order = rank_metrics(dice, c)
                non_scores, _ = rank_metrics(dice[nonempty], c[nonempty])
                ranked_ids = [ids[i] for i in order]
                top59 = set(ranked_ids[:59])
                selected = np.array([cid in top59 for cid in ids])
                zero_top59_count = sum(cid in top59 for cid in zero_nonempty_ids)
                all_scores["dice_zero_nonempty_top59_count"] = int(zero_top59_count)
                non_scores["dice_zero_nonempty_top59_count"] = int(zero_top59_count)
                for subset, scores, n in [("all101", all_scores, len(ids)), ("nonempty81", non_scores, int(nonempty.sum()))]:
                    summary_rows.append({"metric":metric,"alpha":float(alpha),"gating":gating,"subset":subset,"n":n,**scores})
                for group, mask in [("High",selected),("Low",~selected)]:
                    gd = dice[mask]
                    for band, bandmask in [("Dice<0.1",gd<.1),("0.1<=Dice<=0.5",(gd>=.1)&(gd<=.5)),("Dice>0.5",gd>.5)]:
                        bin_rows.append({"metric":metric,"alpha":float(alpha),"gating":gating,"group":group,
                                         "group_n":int(mask.sum()),"dice_band":band,"count":int(bandmask.sum()),
                                         "proportion":float(bandmask.mean()) if len(gd) else np.nan})
                ranks = {cid:i+1 for i,cid in enumerate(ranked_ids)}
                non_ids = [ids[i] for i in np.flatnonzero(nonempty)]
                non_order = np.lexsort((np.arange(len(non_ids)), -c[nonempty]))
                non_ranks = {non_ids[i]:j+1 for j,i in enumerate(non_order)}
                for cid, d, rawc, cval in zip(ids,dice,conf,c):
                    case_rows.append({"case_id":cid,"metric":metric,"alpha":float(alpha),"gating":gating,
                                      "uncertainty":float(u[ids.index(cid)]),"confidence":float(cval),
                                      "raw_confidence":float(rawc),"dice":float(d),"empty_prediction":cid in empty_ids,
                                      "dice_zero_nonempty":cid in zero_nonempty_ids,"rank_all101":ranks[cid],
                                      "rank_nonempty81":non_ranks.get(cid,np.nan),
                                      "top59":cid in top59,"control":"Original E (alpha=0.5)" if alpha==.5 else ("Pure text (alpha=1)" if alpha==1 else "")})
                for cid in sorted(zero_nonempty_ids):
                    i=ids.index(cid)
                    # Compact failure-case audit for the requested 22 cases.
                    if gating == "raw":
                        pass
                # control Original CUE-DIC row, handled after the sweep

    # Add original CUE-DIC control observations, with the same gated-empty confidence=0 convention.
    cue = src[(src.metric == "I0_CUE_DIC") & (src.perturbation == "Original-A")].set_index(["case_id","gating"])
    for gating in ["raw","gated"]:
        for cid, d in zip(ids,dice):
            row = cue.loc[(cid,gating)]
            confidence = float(row.confidence)
            if gating == "gated" and cid in empty_ids: confidence = 0.0
            case_rows.append({"case_id":cid,"metric":"Original_CUE_DIC","alpha":np.nan,"gating":gating,
                              "uncertainty":float(row.uncertainty) if np.isfinite(row.uncertainty) else np.nan,
                              "confidence":confidence,"raw_confidence":float(row.confidence),"dice":float(d),
                              "empty_prediction":cid in empty_ids,"dice_zero_nonempty":cid in zero_nonempty_ids,
                              "rank_all101":np.nan,"rank_nonempty81":np.nan,"top59":False,"control":"Original CUE-DIC"})

    cases = pd.DataFrame(case_rows)
    summaries = pd.DataFrame(summary_rows)
    bins = pd.DataFrame(bin_rows)
    # Full rank table, including ranks and Top-59 membership for the 22 zero-Dice nonempty predictions.
    failure_rows=[]
    for (metric, alpha, gating), grp in cases[cases.metric.isin(METRICS)].groupby(["metric","alpha","gating"]):
        z=grp[grp.dice_zero_nonempty]
        for _,r in z.iterrows():
            failure_rows.append({k:r[k] for k in ["case_id","metric","alpha","gating","confidence","rank_all101","top59","dice","empty_prediction"]})
    failures=pd.DataFrame(failure_rows)

    # Original CUE-DIC control statistics and E routes represented by alpha=.5 (exact previous Image+Prompt scores).
    for gating in ["raw","gated"]:
        g=cases[(cases.metric=="Original_CUE_DIC")&(cases.gating==gating)]
        sm, order=rank_metrics(g.dice.to_numpy(float),g.confidence.to_numpy(float))
        summaries=pd.concat([summaries,pd.DataFrame([{"metric":"Original_CUE_DIC","alpha":np.nan,"gating":gating,"subset":"all101","n":101,**sm}])],ignore_index=True)

    # Paired bootstrap on the full 101 paired cases for the empirically best raw AURC alpha per metric.
    rng=np.random.default_rng(SEED)
    bootstrap_rows=[]
    for metric in METRICS:
        for gating in ["raw","gated"]:
            eligible=summaries[(summaries.metric==metric)&(summaries.gating==gating)&(summaries.subset=="all101")]
            best_alpha=float(eligible.sort_values(["aurc","alpha"]).iloc[0].alpha)
            for comparator in [.5,1.0]:
                a=cases[(cases.metric==metric)&(cases.alpha==best_alpha)&(cases.gating==gating)].sort_values("case_id")
                b=cases[(cases.metric==metric)&(cases.alpha==comparator)&(cases.gating==gating)].sort_values("case_id")
                dd=a.dice.to_numpy(float); ca=a.confidence.to_numpy(float); cb=b.confidence.to_numpy(float)
                for stat, direction in [("spearman","higher"),("pearson","higher"),("aurc","lower"),("top59_mean_dice","higher")]:
                    idxs=rng.integers(0,len(dd),size=(2000,len(dd)))
                    diff=[]
                    for ids_sample in idxs:
                        va=statistic_value(dd[ids_sample],ca[ids_sample],stat)
                        vb=statistic_value(dd[ids_sample],cb[ids_sample],stat)
                        if np.isfinite(va) and np.isfinite(vb): diff.append(va-vb)
                    diff=np.asarray(diff)
                    bootstrap_rows.append({"metric":metric,"gating":gating,"best_alpha_by_gating_AURC":best_alpha,
                                           "comparator_alpha":comparator,"statistic":stat,"direction_better":direction,
                                           "point_difference":float((rank_metrics(dd,ca)[0][stat]-rank_metrics(dd,cb)[0][stat])),
                                           "ci_low":float(np.quantile(diff,.025)),"ci_high":float(np.quantile(diff,.975)),
                                           "bootstrap_replicates":2000,"seed":SEED,
                                           "improvement_stable":bool(np.quantile(diff,.025)>0 if direction=="higher" else np.quantile(diff,.975)<0)})
    boot=pd.DataFrame(bootstrap_rows)

    cases.to_csv(outdir/"per_case_scores.csv",index=False)
    summaries.to_csv(outdir/"fusion_summary.csv",index=False)
    bins.to_csv(outdir/"high_low_dice_bins.csv",index=False)
    failures.to_csv(outdir/"zero_dice_nonempty_cases.csv",index=False)
    boot.to_csv(outdir/"paired_bootstrap_comparison.csv",index=False)
    anchor_rows=[]
    for m in METRICS:
        ss=metric_rows[metric_rows.metric==m]
        for a, perturb in [(0.0,"Image-only"),(.5,"Image+Prompt"),(1.0,"Prompt-only")]:
            for gate in ["raw","gated"]:
                new=cases[(cases.metric==m)&(cases.alpha==a)&(cases.gating==gate)].set_index("case_id").loc[ids]
                old=ss[(ss.perturbation==perturb)&(ss.gating==gate)].set_index("case_id").loc[ids]
                comparable=np.array([cid not in empty_ids or gate=="raw" for cid in ids])
                diff=np.abs(new.confidence.to_numpy(float)[comparable]-old.confidence.to_numpy(float)[comparable])
                anchor_rows.append({"metric":m,"alpha":a,"existing_source":perturb,"gating":gate,
                                    "comparable_cases":int(comparable.sum()),"max_abs_confidence_difference":float(diff.max()),
                                    "empty_cases_set_to_zero":int(len(empty_ids) if gate=="gated" else 0)})
                assert diff.max()<1e-10
    pd.DataFrame(anchor_rows).to_csv(outdir/"alpha_anchor_validation.csv",index=False)

    # Curves: raw/gated are separate lines; full cohort and nonempty Spearman are explicit.
    colors={m:c for m,c in zip(METRICS,["#2864a5","#db7c26","#26945b"])}
    fig,axs=plt.subplots(1,3,figsize=(15,4),sharex=True)
    for ax,m in zip(axs,METRICS):
        for gate,ls in [("raw","-"),("gated","--")]:
            z=summaries[(summaries.metric==m)&(summaries.gating==gate)]
            for subset,marker in [("all101","o"),("nonempty81","s")]:
                q=z[z.subset==subset].sort_values("alpha")
                ax.plot(q.alpha,q.spearman,linestyle=ls,marker=marker,color=colors[m],
                        label=f"{gate} {subset}")
        ax.set(title=m,xlabel="Text uncertainty weight α",ylabel="Spearman")
        ax.grid(alpha=.25)
    axs[-1].legend(fontsize=7,bbox_to_anchor=(1.02,1),loc="upper left")
    fig.tight_layout();fig.savefig(figdir/"spearman_vs_alpha.png",dpi=180,bbox_inches="tight");plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(15,4),sharex=True)
    for ax,m in zip(axs,METRICS):
        for gate,ls in [("raw","-"),("gated","--")]:
            q=summaries[(summaries.metric==m)&(summaries.gating==gate)&(summaries.subset=="all101")].sort_values("alpha")
            ax.plot(q.alpha,q.aurc,linestyle=ls,marker="o",color=colors[m],label=gate)
        ax.set(title=m,xlabel="Text uncertainty weight α",ylabel="AURC (lower is better)");ax.grid(alpha=.25)
    axs[-1].legend();fig.tight_layout();fig.savefig(figdir/"aurc_vs_alpha.png",dpi=180);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(16,4.5),sharex=True)
    for ax,m in zip(axs,METRICS):
        for gate,ls in [("raw","-"),("gated","--")]:
            z=summaries[(summaries.metric==m)&(summaries.gating==gate)&(summaries.subset=="all101")].sort_values("alpha")
            for k in TOPKS:
                ax.plot(z.alpha,z[f"top{k}_mean_dice"],linestyle=ls,marker=".",label=f"{gate} Top-{k}")
        ax.set(title=m,xlabel="Text uncertainty weight α",ylabel="Top-K mean Dice");ax.grid(alpha=.25)
    axs[-1].legend(fontsize=6,ncol=2,bbox_to_anchor=(1.02,1),loc="upper left")
    fig.tight_layout();fig.savefig(figdir/"topk_mean_dice_vs_alpha.png",dpi=180,bbox_inches="tight");plt.close(fig)

    cmap=plt.get_cmap("viridis",len(ALPHAS))
    for gate in ["raw","gated"]:
        fig,axs=plt.subplots(1,3,figsize=(15,4),sharex=True,sharey=True)
        for ax,m in zip(axs,METRICS):
            z=cases[(cases.metric==m)&(cases.gating==gate)]
            for j,a in enumerate(ALPHAS):
                q=z[z.alpha==a]
                ax.scatter(q.confidence,q.dice,s=12,alpha=.45,color=cmap(j),label=f"α={a:.1f}")
            ax.set(title=m,xlabel="Confidence",ylabel="Dice");ax.grid(alpha=.2)
        axs[-1].legend(fontsize=7,ncol=2,bbox_to_anchor=(1.02,1),loc="upper left")
        fig.suptitle(f"Confidence–Dice by fusion weight ({gate})",y=1.02)
        fig.tight_layout();fig.savefig(figdir/f"confidence_dice_scatter_{gate}.png",dpi=180,bbox_inches="tight");plt.close(fig)

    # Report summary table: key raw/gated metrics and bootstrap results.
    bestlines=[]
    raw_candidates={}
    gated_candidates={}
    for metric in METRICS:
        for gate in ["raw","gated"]:
            q=summaries[(summaries.metric==metric)&(summaries.gating==gate)&(summaries.subset=="all101")]
            best=q.sort_values(["aurc","alpha"]).iloc[0]
            bestlines.append(f"| {metric} | {gate} | {best.alpha:.1f} | {best.aurc:.4f} | {best.spearman:.4f} | {best.top59_mean_dice:.4f} |")
            (raw_candidates if gate=="raw" else gated_candidates)[metric]=best
    bootlines=[]
    for _,r in boot.iterrows():
        bootlines.append(f"| {r.metric} | {r.gating} | {r.best_alpha_by_gating_AURC:.1f} | {r.comparator_alpha:.1f} | {r.statistic} | {r.point_difference:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] | {'是' if r.improvement_stable else '否'} |")
    raw_improvements=[]
    for metric,best in raw_candidates.items():
        q=summaries[(summaries.metric==metric)&(summaries.gating=="raw")&(summaries.subset=="all101")].set_index("alpha")
        raw_improvements.append(f"{metric}: α={best.alpha:.1f} AURC {best.aurc:.4f}, vs image-only α=0 {best.aurc-q.loc[0.0,'aurc']:+.4f}, vs pure text α=1 {best.aurc-q.loc[1.0,'aurc']:+.4f}")
    gated_alpha_text=", ".join(f"{m} α={r.alpha:.1f}" for m,r in gated_candidates.items())
    report=f'''# Image/Prompt uncertainty fusion weight ablation

## Scope and method

This is an offline exploratory sweep using only the existing `per_case_scores.csv`; there was no inference, model training, parameter update, or adaptation. The 101 cases and their GT Dice values come from the current benchmark. Therefore, alpha selection and all comparisons are exploratory and use training-set GT; they are not an unbiased held-out evaluation. This sweep does not resolve the upstream historical baseline discrepancy: the prior audit records a mean absolute Dice difference of about `6e-4`, which remained unexplained after checking with the repository's original `VoxTellPredictor`. Treat these results as conditional comparisons within the current score table, not as a validated reproduction of the historical experiment.

For each metric, fusion was performed on the uncertainty values, `U_joint=(1−α)U_image+αU_prompt`, for α=0.0…1.0 by 0.1. Confidence was then recomputed using the existing metric transform: `1−U_joint` for I1/I2 and `1/(1+U_joint)` for I3 VVC. Gated results set confidence to 0 for empty predictions; raw results retain the raw confidence. There are 101 total cases, 81 nonempty predictions, 20 empty predictions, and 22 zero-Dice nonempty predictions. High is the top 59 by confidence, Low the remaining 42; ties are broken by case ID. Dice bands are `<0.1`, `[0.1,0.5]`, and `>0.5`.

## Consistency checks and controls

At α=0, 0.5, and 1.0, scores reproduce the existing Image-only, Image+Prompt, and Prompt-only confidence transforms to numerical precision (see `alpha_anchor_validation.csv`). α=0.5 is the existing E-style equal fusion control; α=1.0 is pure text. Original CUE-DIC is retained separately as a routing control in `per_case_scores.csv` and `fusion_summary.csv`. Gated empty predictions use the requested confidence 0 convention.

## Best AURC candidate by metric and gating

| Metric | Gating | α | AURC | Spearman (101) | Top-59 mean Dice |
|---|---|---:|---:|---:|---:|
{chr(10).join(bestlines)}

Full and nonempty Spearman, Pearson, AURC, AUROC/AUPRC, all requested Top-K means, and High/Low Dice are in `fusion_summary.csv`. The three Dice-band counts and within-group proportions are in `high_low_dice_bins.csv`. Confidence, cohort rank, Top-59 membership, and the 22 zero-Dice nonempty cases are in `per_case_scores.csv` and `zero_dice_nonempty_cases.csv`.

## Paired bootstrap comparisons

For each metric and gating mode, the candidate α minimizing observed all-case AURC was compared with α=0.5 and α=1.0 using 2,000 paired case bootstrap resamples. The intervals quantify uncertainty conditional on this cohort and the selected candidate; because candidate selection and assessment use the same GT cohort, they do not establish generalization.

| Metric | Gating | Candidate α | Comparator α | Statistic (candidate−comparator) | 95% paired CI | Stable improvement |
|---|---|---:|---:|---|---:|---|
{chr(10).join(bootlines)}

## Interpretation

The raw AURC-selected interior mixtures improve point estimates over both single-source endpoints for all three metrics: {"; ".join(raw_improvements)}. This supports image/text complementarity in this cohort. Against pure text α=1, raw paired-bootstrap AURC and Spearman improvements are stable for the selected candidates, while Top-59 Dice gains are not stable. Against equal fusion α=0.5, raw AURC differences are not stable for I1 and I3; I2 selects α=0.5 itself. In gated results the AURC-minimizing weights are {gated_alpha_text}; only VVC at α=0.9 has a stable AURC improvement over gated α=0.5 in this sweep. Thus increasing text weight above 0.5 does not give a consistent improvement across metrics and gating modes.

The most defensible weight to retain is α=0.5: it is the raw AURC optimum for I2, lies close to the raw optima for I1 and I3 without statistically stable gains over 0.5, and is the existing equal-fusion E control. Higher text weights remain candidates for a held-out check, particularly with gating, but are not established as a universal replacement. All these conclusions are exploratory because alpha was selected and evaluated against the same training-set GT cohort.

## Figures

`figures/spearman_vs_alpha.png`, `figures/aurc_vs_alpha.png`, and `figures/topk_mean_dice_vs_alpha.png` show the alpha curves. `figures/confidence_dice_scatter_raw.png` and `figures/confidence_dice_scatter_gated.png` show Confidence–Dice distributions for all fusion weights.
'''
    (outdir/"REPORT.md").write_text(report)
    print(f"Wrote ablation to {outdir}; {len(cases)} case rows, {len(summaries)} summary rows, {len(boot)} bootstrap comparisons")


if __name__ == "__main__":
    main()
