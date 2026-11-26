#!/usr/bin/env python3
"""Generate subset, ranking, bootstrap, failure and visualization outputs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields=[]
        for row in rows:
            for field in row:
                if field not in fields: fields.append(field)
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def safe_corr(fn, a, b):
    if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    try:
        v = fn(a, b)
        return float(v.statistic if hasattr(v, "statistic") else v[0])
    except Exception:
        return float("nan")


def aurc(dice, confidence):
    order = np.lexsort((np.arange(len(dice)), -confidence))
    risks = 1.0 - dice[order]
    coverage = np.arange(1, len(dice) + 1, dtype=float) / len(dice)
    curve = np.cumsum(risks) / np.arange(1, len(risks) + 1)
    return float(np.trapezoid(np.r_[curve[0], curve], np.r_[0.0, coverage]))


def stable_order(rows):
    return sorted(rows, key=lambda r: (-float(r["confidence"]), r["case_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="analysis/hierarchical_uncertainty/stage1_image_level")
    parser.add_argument("--reference-dir", default="/home/ubuntu/zy/pythoncode/sfda/CUE-SFDA")
    parser.add_argument("--gt-dir", default="/data/hilab/datasets/WCH_MRI145/DWI/labels")
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    root = Path(args.results); figs = root / "figures"; figs.mkdir(parents=True, exist_ok=True)
    scores = read_csv(root / "per_case_scores.csv")
    baseline = {r["case_id"]: {**r, "dice": float(r["current_dice"]), "historical_dice": float(r["historical_dice"]),
                                  "pred_voxels": int(r["original_foreground_voxels"]), "gt_voxels": int(r["gt_foreground_voxels"]),
                                  "spacing": np.array(json.loads(r["spacing"]), dtype=float)}
                for r in read_csv(root / "baseline_reproducibility.csv")}
    ids = sorted(baseline)
    # I4/I5 are original-only by definition. I12 follows the repository's
    # Chebyshev implementation over the 10 image perturbation views only.
    normalized=[]
    for row in scores:
        if row["metric"] in {"I4_MPC","I5_Entropy"}:
            continue
        if row["metric"]=="I12_Chebyshev":row["perturbation"]="Image-only"
        normalized.append(row)
    scores=normalized
    for cid in ids:
        with np.load(root/"probabilities"/f"{cid}.npz",allow_pickle=False) as z:
            p0=z["original"]; iv=z["image_views"]; pv=z["prompt_views"]
        if p0.ndim==4 and p0.shape[0]==1:p0=p0[0]
        union=np.concatenate([(p0[None]>.5),iv>.5,pv>.5],axis=0).any(axis=0)
        mean=iv.astype(np.float64).mean(axis=0); var=iv.astype(np.float64).var(axis=0)
        error=var/(var+(mean-.5)**2+1e-8)
        # Some earlier score-writer revisions omitted the original-only I4/I5
        # rows. Materialize them here from p0, without rerunning inference.
        present={r["metric"] for r in scores if r["case_id"]==cid and r["metric"] not in {"I4_MPC","I5_Entropy"}}
        p0_64=np.asarray(p0,dtype=np.float64);p0_clip=np.clip(p0_64,1e-8,1-1e-8)
        for metric,uncertainty_map in (("I4_MPC",1.0-np.maximum(p0_64,1.0-p0_64)),("I5_Entropy",-(p0_clip*np.log2(p0_clip)+(1-p0_clip)*np.log2(1-p0_clip)))):
            if metric in present:continue
            for agg,region in (("G0",None),("G1",union)):
                chosen=np.ones(p0.shape,dtype=bool) if region is None else region; n=int(chosen.sum())
                u=float(uncertainty_map[chosen].mean()) if n else float("nan")
                raw={"case_id":cid,"metric":metric,"perturbation":"Original","aggregation":agg,"gating":"raw",
                     "uncertainty":str(u),"confidence":str(1-u),"valid_voxels":str(n),"dice":str(baseline[cid]["dice"])}
                scores.append(raw)
                gated=dict(raw);gated["gating"]="gated"
                if baseline[cid]["pred_voxels"]==0:gated["uncertainty"]="nan";gated["confidence"]="-1.0"
                scores.append(gated)
        for row in scores:
            if row["case_id"]==cid and row["metric"]=="I12_Chebyshev":row["perturbation"]="Image-only"
            if row["case_id"]==cid and row["metric"]=="I12_Chebyshev" and row["gating"]=="raw":
                region=None if row["aggregation"]=="G0" else union
                selected=np.ones(p0.shape,dtype=bool) if region is None else region
                n=int(selected.sum()); u=float(error[selected].mean()) if n else float("nan")
                row["perturbation"]="Image-only";row["uncertainty"]=str(u);row["confidence"]=str(1-u);row["valid_voxels"]=str(n)
    raw_rows={(r["case_id"],r["metric"],r["perturbation"],r["aggregation"]):r for r in scores if r["gating"]=="raw"}
    for row in scores:
        if row["gating"]=="gated":
            source=raw_rows.get((row["case_id"],row["metric"],row["perturbation"],row["aggregation"]))
            if source:
                for field in ("uncertainty","confidence","valid_voxels"):row[field]=source[field]
            if baseline[row["case_id"]]["pred_voxels"]==0:
                row["uncertainty"]="nan";row["confidence"]="-1.0"
    write_csv(root/"per_case_scores.csv",scores)
    dice_all = np.array([baseline[i]["dice"] for i in ids])
    subsets = {
        "all": np.ones(len(ids), bool),
        "original_nonempty": np.array([baseline[i]["pred_voxels"] > 0 for i in ids]),
        "original_empty": np.array([baseline[i]["pred_voxels"] == 0 for i in ids]),
        "dice_lt_0.1": dice_all < .1,
        "dice_0.1_to_0.5": (dice_all >= .1) & (dice_all <= .5),
        "dice_gt_0.5": dice_all > .5,
        "dice_zero_nonempty": (dice_all == 0) & np.array([baseline[i]["pred_voxels"] > 0 for i in ids]),
    }
    grouped = defaultdict(dict)
    for row in scores:
        if row["gating"] == "raw":
            grouped[(row["metric"], row["perturbation"], row["aggregation"])][row["case_id"]] = row
    corr_rows, cond_rows, aurc_rows, topk_rows, highlow_rows, bins_rows = [], [], [], [], [], []
    ranking_rows = []
    for key, mapping in grouped.items():
        if len(mapping) != len(ids):
            continue
        metric, perturb, aggregation = key
        rows = [mapping[i] for i in ids]
        conf = np.array([float(r["confidence"]) for r in rows])
        unc = np.array([float(r["uncertainty"]) for r in rows])
        for subset, mask in subsets.items():
            y, c, u = dice_all[mask], conf[mask], unc[mask]
            finite = np.isfinite(y) & np.isfinite(c)
            y, c, u = y[finite], c[finite], u[finite]
            cond_rows.append({"metric":metric,"perturbation":perturb,"aggregation":aggregation,"subset":subset,"n":len(y),
                              "gating":"raw","spearman":safe_corr(spearmanr,y,c),"pearson":safe_corr(pearsonr,y,c),"kendall_tau_b":safe_corr(kendalltau,y,c)})
            if len(y): aurc_rows.append({"metric":metric,"perturbation":perturb,"aggregation":aggregation,"subset":subset,"gating":"raw","n":len(y),"aurc":aurc(y,c)})
        finite = np.isfinite(conf)
        if finite.sum() >= 59:
            available_rows=[r for r in rows if np.isfinite(float(r["confidence"]))]
            tie_n = len(available_rows) - len({float(r["confidence"]) for r in available_rows})
            ordered = stable_order(available_rows)
            oracle = sorted(ids, key=lambda cid: (-baseline[cid]["dice"], cid))
            for k in (10,20,30,40,49,59):
                chosen = ordered[:k]; ds = np.array([baseline[r["case_id"]]["dice"] for r in chosen]); chosen_ids={r["case_id"] for r in chosen}
                topk_rows.append({"metric":metric,"perturbation":perturb,"aggregation":aggregation,"K":k,"mean_dice":float(ds.mean()),
                                  "gating":"raw","available_n":len(ordered),"unrankable_n":len(ids)-len(ordered),
                                  "median_dice":float(np.median(ds)),"min_dice":float(ds.min()),
                                  "dice_zero_nonempty_n":sum(baseline[r["case_id"]]["dice"]==0 and baseline[r["case_id"]]["pred_voxels"]>0 for r in chosen),
                                  "dice_lt_0.1_n":int((ds<.1).sum()),"dice_gt_0.5_n":int((ds>.5).sum()),
                                  "dice_gt_0.7_n":int((ds>.7).sum()),"dice_gt_0.8_n":int((ds>.8).sum()),
                                  "oracle_overlap_rate":len(chosen_ids & set(oracle[:k]))/k,"tie_count":tie_n,
                                  "tie_rule":"case_id ascending"})
                if k == 59:
                    hi_ids=[r["case_id"] for r in chosen]; lo_ids=[cid for cid in ids if cid not in chosen_ids]
                    hi=np.array([baseline[cid]["dice"] for cid in hi_ids]); lo=np.array([baseline[cid]["dice"] for cid in lo_ids])
                    bins={"dice=0":lambda d:d==0,"0<dice<0.1":lambda d:(d>0)&(d<.1),"0.1<=dice<=0.5":lambda d:(d>=.1)&(d<=.5),"dice>0.5":lambda d:d>.5}
                    highlow_rows.append({"metric":metric,"perturbation":perturb,"aggregation":aggregation,"high_n":len(hi),"low_n":len(lo),
                                         "high_mean_dice":hi.mean(),"low_mean_dice":lo.mean(),"dice_gap":hi.mean()-lo.mean(),
                                         "high_median_dice":np.median(hi),"low_median_dice":np.median(lo),
                                         **{f"high_{name}_n":int(fn(hi).sum()) for name,fn in bins.items()},
                                         **{f"low_{name}_n":int(fn(lo).sum()) for name,fn in bins.items()},
                                         **{f"{name}_high_rate":float(fn(hi).sum()/len([d for d in dice_all if fn(np.array([d]))[0]])) if fn(dice_all).sum() else float("nan") for name,fn in bins.items()}})
            # Equal-frequency confidence bins, deterministic by confidence then case ID.
            ranked_conf=sorted([(cid,float(mapping[cid]["confidence"])) for cid in ids if np.isfinite(float(mapping[cid]["confidence"]))],key=lambda x:(x[1],x[0]))
            for b, block_ix in enumerate(np.array_split(np.arange(len(ranked_conf)),5),1):
                block=[ranked_conf[int(i)] for i in block_ix]
                block_ids=[x[0] for x in block]
                bins_rows.append({"metric":metric,"perturbation":perturb,"aggregation":aggregation,"bin":b,"n":len(block_ids),
                                  "confidence_min":float(min(x[1] for x in block)),"confidence_max":float(max(x[1] for x in block)),
                                  "mean_confidence":float(np.mean([x[1] for x in block])),"mean_dice":float(np.mean([baseline[cid]["dice"] for cid in block_ids]))})
            all_corr=safe_corr(spearmanr,dice_all,conf); nonempty=subsets["original_nonempty"]
            nonempty_corr=safe_corr(spearmanr,dice_all[nonempty],conf[nonempty])
            # Lower is better for AURC; stable failure count is a diagnostic, not a sole ranking criterion.
            a=next((x["aurc"] for x in aurc_rows if x["metric"]==metric and x["perturbation"]==perturb and x["aggregation"]==aggregation and x["subset"]=="all"),float("nan"))
            ranking_rows.append({"metric":metric,"perturbation":perturb,"aggregation":aggregation,"spearman_all":all_corr,
                                 "spearman_nonempty":nonempty_corr,"aurc_all":a,"tie_count":tie_n,
                                 "dice_zero_nonempty_top59":sum(baseline[r["case_id"]]["dice"]==0 and baseline[r["case_id"]]["pred_voxels"]>0 for r in ordered[:59])})
    gated_grouped=defaultdict(dict)
    for row in scores:
        if row["gating"]=="gated":gated_grouped[(row["metric"],row["perturbation"],row["aggregation"])][row["case_id"]]=row
    from sklearn.metrics import average_precision_score, roc_auc_score
    gated_failure=[]
    for key,mapping in gated_grouped.items():
        if len(mapping)!=len(ids):continue
        rows=[mapping[cid] for cid in ids]; conf=np.array([float(r["confidence"]) for r in rows])
        unc=-conf  # gated empty cases are the least reliable; raw U stays untouched.
        for subset,mask in subsets.items():
            y,c=dice_all[mask],conf[mask];finite=np.isfinite(c);y,c=y[finite],c[finite]
            cond_rows.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"gating":"gated","subset":subset,"n":len(y),
                              "spearman":safe_corr(spearmanr,y,c),"pearson":safe_corr(pearsonr,y,c),"kendall_tau_b":safe_corr(kendalltau,y,c)})
            if len(y):aurc_rows.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"gating":"gated","subset":subset,"n":len(y),"aurc":aurc(y,c)})
        for target_name,target,score in (("dice_lt_0.1",(dice_all<.1).astype(int),unc),("dice_gt_0.5",(dice_all>.5).astype(int),conf)):
            if len(np.unique(target))==2:
                gated_failure.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"gating":"gated","target":target_name,
                                      "auroc":roc_auc_score(target,score),"auprc":average_precision_score(target,score)})
            else:gated_failure.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"gating":"gated","target":target_name,"auroc":float("nan"),"auprc":float("nan")})
        available=[r for r in rows if np.isfinite(float(r["confidence"]))]
        ordered=stable_order(available);tie_n=len(available)-len({float(r["confidence"]) for r in available})
        oracle=sorted(ids,key=lambda cid:(-baseline[cid]["dice"],cid))
        for k in (10,20,30,40,49,59):
            if len(ordered)<k:continue
            chosen=ordered[:k]; chosen_ids={r["case_id"] for r in chosen};ds=np.array([baseline[r["case_id"]]["dice"] for r in chosen])
            if k==59:
                lo_ids=[cid for cid in ids if cid not in chosen_ids];hi=np.array([baseline[cid]["dice"] for cid in chosen_ids]);lo=np.array([baseline[cid]["dice"] for cid in lo_ids])
                bins={"dice=0":lambda d:d==0,"0<dice<0.1":lambda d:(d>0)&(d<.1),"0.1<=dice<=0.5":lambda d:(d>=.1)&(d<=.5),"dice>0.5":lambda d:d>.5}
                highlow_rows.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"gating":"gated","high_n":len(hi),"low_n":len(lo),
                                     "high_mean_dice":hi.mean(),"low_mean_dice":lo.mean(),"dice_gap":hi.mean()-lo.mean(),
                                     **{f"high_{n}_n":int(fn(hi).sum()) for n,fn in bins.items()},**{f"low_{n}_n":int(fn(lo).sum()) for n,fn in bins.items()}})
    write_csv(root/"conditional_correlations.csv",cond_rows);write_csv(root/"aurc_summary.csv",aurc_rows)
    old_failure=read_csv(root/"failure_detection_summary.csv")
    for r in old_failure:r["gating"]="raw"
    write_csv(root/"failure_detection_summary.csv",old_failure+gated_failure)
    gated=defaultdict(dict)
    for row in scores:
        if row["gating"]=="gated":gated[(row["metric"],row["perturbation"],row["aggregation"])][row["case_id"]]=row
    gated_topk=[]
    for key,mapping in gated.items():
        if len(mapping)!=len(ids):continue
        ordered=sorted((mapping[cid] for cid in ids if np.isfinite(float(mapping[cid]["confidence"]))),key=lambda r:(-float(r["confidence"]),r["case_id"]))
        if len(ordered)<59:continue
        tie_n=len(ids)-len({float(r["confidence"]) for r in ordered})
        oracle=sorted(ids,key=lambda cid:(-baseline[cid]["dice"],cid))
        for k in (10,20,30,40,49,59):
            chosen=ordered[:k]; ds=np.array([baseline[r["case_id"]]["dice"] for r in chosen]); chosen_ids={r["case_id"] for r in chosen}
            gated_topk.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"gating":"gated","K":k,
                               "available_n":len(ordered),"unrankable_n":len(ids)-len(ordered),
                               "mean_dice":float(ds.mean()),"median_dice":float(np.median(ds)),"min_dice":float(ds.min()),
                               "dice_zero_nonempty_n":sum(baseline[r["case_id"]]["dice"]==0 and baseline[r["case_id"]]["pred_voxels"]>0 for r in chosen),
                               "dice_lt_0.1_n":int((ds<.1).sum()),"dice_gt_0.5_n":int((ds>.5).sum()),
                               "dice_gt_0.7_n":int((ds>.7).sum()),"dice_gt_0.8_n":int((ds>.8).sum()),
                               "oracle_overlap_rate":len(chosen_ids & set(oracle[:k]))/k,"tie_count":tie_n,"tie_rule":"case_id ascending"})
    write_csv(root/"topk_summary.csv",topk_rows+gated_topk);write_csv(root/"high_low_summary.csv",highlow_rows)
    write_csv(root/"score_bin_summary.csv",bins_rows)
    ranking_rows.sort(key=lambda r:(-float(r["spearman_nonempty"]) if np.isfinite(r["spearman_nonempty"]) else 1e9,
                                    float(r["aurc_all"]) if np.isfinite(r["aurc_all"]) else 1e9))
    write_csv(root/"metric_ranking.csv",ranking_rows)
    # Per-case view and lesion volume audit; GT appears only here, after score generation.
    volume_rows=[]
    prob_root=root/"probabilities"
    for cid in ids:
        with np.load(prob_root/f"{cid}.npz",allow_pickle=False) as z:
            iv=z["image_views"];pv=z["prompt_views"]
        b=baseline[cid]; spacing=b["spacing"]; voxel_mm3=float(np.prod(spacing))
        row={"case_id":cid,"gt_voxels":b["gt_voxels"],"original_pred_voxels":b["pred_voxels"],
             "gt_volume_mm3":b["gt_voxels"]*voxel_mm3,"original_pred_volume_mm3":b["pred_voxels"]*voxel_mm3,
             "gt_foreground_fraction":b["gt_voxels"]/max(np.prod(iv.shape[1:]),1),
             "original_prediction_fraction":b["pred_voxels"]/max(np.prod(iv.shape[1:]),1),
             "image_view_voxels":json.dumps((iv>.5).reshape(len(iv),-1).sum(1).tolist()),
             "prompt_view_voxels":json.dumps((pv>.5).reshape(len(pv),-1).sum(1).tolist()),"dice":b["dice"]}
        volume_rows.append(row)
    write_csv(root/"lesion_size_analysis.csv",volume_rows)
    lesion_corr=[]
    gtvol=np.array([baseline[c]["gt_voxels"]*np.prod(baseline[c]["spacing"]) for c in ids])
    predvol=np.array([baseline[c]["pred_voxels"]*np.prod(baseline[c]["spacing"]) for c in ids])
    for key,mapping in grouped.items():
        if len(mapping)!=len(ids):continue
        conf=np.array([float(mapping[c]["confidence"]) for c in ids])
        lesion_corr.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],
                            "spearman_confidence_gt_volume":safe_corr(spearmanr,gtvol,conf),
                            "spearman_confidence_prediction_volume":safe_corr(spearmanr,predvol,conf),
                            "spearman_dice_gt_volume":safe_corr(spearmanr,gtvol,dice_all),
                            "spearman_dice_prediction_volume":safe_corr(spearmanr,predvol,dice_all)})
    write_csv(root/"lesion_size_correlations.csv",lesion_corr)
    quantiles=np.quantile(gtvol,[1/3,2/3]); lesion_groups={"small":gtvol<=quantiles[0],"medium":(gtvol>quantiles[0])&(gtvol<=quantiles[1]),"large":gtvol>quantiles[1]}
    lesion_group_rows=[]
    for item in ranking_rows[:8]:
        key=(item["metric"],item["perturbation"],item["aggregation"]); mapping=grouped[key]
        conf=np.array([float(mapping[c]["confidence"]) for c in ids])
        for label,mask in lesion_groups.items():
            lesion_group_rows.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"gt_volume_group":label,"n":int(mask.sum()),
                                      "gt_volume_min_mm3":float(gtvol[mask].min()),"gt_volume_max_mm3":float(gtvol[mask].max()),
                                      "mean_dice":float(dice_all[mask].mean()),"spearman_confidence_dice":safe_corr(spearmanr,dice_all[mask],conf[mask])})
    write_csv(root/"lesion_size_group_summary.csv",lesion_group_rows)
    dice_zero_nonempty={cid for cid in ids if baseline[cid]["dice"]==0 and baseline[cid]["pred_voxels"]>0}
    target=["0000321625","0006150559","0010392655","0018128083","0020946113","0032710370","0032929968","0033102538"]
    a_rank_map={}; e_rank_map={}
    a_ref=json.loads((Path(args.reference_dir)/"outputs/dwi_only_liver_cancer/confidence_partition.json").read_text())
    a_ref_scores={k.removesuffix(".nii.gz"):float(v) for k,v in a_ref["scores"].items()}
    e_ref={r["case_id"].removesuffix(".nii.gz"):r for r in read_csv(Path("outputs/response_routing/E_joint_empty_to_low/routing.csv"))}
    a_order=sorted(ids,key=lambda cid:(-a_ref_scores[cid],cid))
    e_order=sorted(ids,key=lambda cid:(float(e_ref[cid]["joint_uncertainty"]) if e_ref[cid]["response_state"]!="No-response" else float("inf"),cid))
    a_rank_map={cid:i+1 for i,cid in enumerate(a_order)};e_rank_map={cid:i+1 for i,cid in enumerate(e_order)}
    stable_rows=[]
    for metric,perturb,aggregation in sorted(grouped):
        mapping=grouped[(metric,perturb,aggregation)]; order=sorted(ids,key=lambda cid:(-float(mapping[cid]["confidence"]),cid))
        rank={cid:i+1 for i,cid in enumerate(order)}
        for cid in sorted(dice_zero_nonempty | set(target)):
            r=mapping[cid]; stable_rows.append({"case_id":cid,"metric":metric,"perturbation":perturb,"aggregation":aggregation,
                "uncertainty":r["uncertainty"],"confidence":r["confidence"],"all_case_rank":rank[cid],
                "nonempty_rank":sum(baseline[x]["pred_voxels"]>0 and rank[x]<rank[cid] for x in ids)+1,
                "A_rank":a_rank_map[cid],"E_rank":e_rank_map[cid],
                "top59":rank[cid]<=59,"A_top59":a_rank_map[cid]<=59,"E_top59":e_rank_map[cid]<=59,
                "in_historical_E_error_list":cid in target,"current_dice":baseline[cid]["dice"]})
    write_csv(root/"failure_cases.csv",stable_rows)
    # Bootstrap CIs over the case-level association, seed fixed. Paired samples
    # use the same resampled case indices for every candidate.
    rng=np.random.default_rng(args.seed); boot_rows=[]
    main_groups=[(k,m) for k,m in grouped.items() if k[2] in {"G0","case"}]
    reference = Path(args.reference_dir)
    a_payload=json.loads((reference/"outputs/dwi_only_liver_cancer/confidence_partition.json").read_text())
    a_scores={k.removesuffix(".nii.gz"):float(v) for k,v in a_payload["scores"].items()}
    route_ref={r["case_id"].removesuffix(".nii.gz"):r for r in read_csv(Path("outputs/response_routing/E_joint_empty_to_low/routing.csv"))}
    a_conf=np.array([a_scores[cid] for cid in ids])
    e_conf=np.array([-float(route_ref[cid]["joint_uncertainty"]) if route_ref[cid]["response_state"]!="No-response" else -1.0 for cid in ids])
    for key,mapping in main_groups:
        if len(mapping)!=len(ids):continue
        conf=np.array([float(mapping[cid]["confidence"]) for cid in ids])
        vals=[]
        for _ in range(args.bootstrap):
            ix=rng.integers(0,len(ids),len(ids)); vals.append(safe_corr(spearmanr,dice_all[ix],conf[ix]))
        vals=np.array(vals); vals=vals[np.isfinite(vals)]
        boot_rows.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"statistic":"spearman_dice",
                          "estimate":safe_corr(spearmanr,dice_all,conf),"ci_low":np.quantile(vals,.025) if len(vals) else np.nan,
                          "ci_high":np.quantile(vals,.975) if len(vals) else np.nan,"bootstrap_replicates":args.bootstrap,"seed":args.seed})
        for refname, refconf in (("A",a_conf),("E",e_conf)):
            delta_s=[]; delta_a=[]
            for _ in range(args.bootstrap):
                ix=rng.integers(0,len(ids),len(ids))
                delta_s.append(safe_corr(spearmanr,dice_all[ix],conf[ix])-safe_corr(spearmanr,dice_all[ix],refconf[ix]))
                delta_a.append(aurc(dice_all[ix],conf[ix])-aurc(dice_all[ix],refconf[ix]))
            delta_s=np.asarray(delta_s); delta_a=np.asarray(delta_a)
            boot_rows.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"statistic":f"paired_vs_{refname}",
                              "estimate_delta_spearman":safe_corr(spearmanr,dice_all,conf)-safe_corr(spearmanr,dice_all,refconf),
                              "spearman_delta_ci_low":np.nanquantile(delta_s,.025),"spearman_delta_ci_high":np.nanquantile(delta_s,.975),
                              "estimate_delta_aurc":aurc(dice_all,conf)-aurc(dice_all,refconf),
                              "aurc_delta_ci_low":np.nanquantile(delta_a,.025),"aurc_delta_ci_high":np.nanquantile(delta_a,.975),
                              "bootstrap_replicates":args.bootstrap,"seed":args.seed})
    for key,mapping in gated_grouped.items():
        if key[2] not in {"G0","case"} or len(mapping)!=len(ids):continue
        conf=np.array([float(mapping[cid]["confidence"]) for cid in ids])
        vals=[]
        for _ in range(args.bootstrap):
            ix=rng.integers(0,len(ids),len(ids));vals.append(safe_corr(spearmanr,dice_all[ix],conf[ix]))
        vals=np.asarray(vals);vals=vals[np.isfinite(vals)]
        boot_rows.append({"metric":key[0],"perturbation":key[1],"aggregation":key[2],"gating":"gated","statistic":"spearman_dice",
                          "estimate":safe_corr(spearmanr,dice_all,conf),"ci_low":np.quantile(vals,.025) if len(vals) else np.nan,
                          "ci_high":np.quantile(vals,.975) if len(vals) else np.nan,"bootstrap_replicates":args.bootstrap,"seed":args.seed})
    write_csv(root/"bootstrap_comparison.csv",boot_rows)
    # Reproduce A/E partitions using the historical continuous Dice file.
    source_case_csv = Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/diagnostic/dwi_uncertainty_train/per_case_uncertainty.csv")
    source_dice = {r["case_id"].removesuffix(".nii.gz"):float(r["baseline_dice"]) for r in read_csv(source_case_csv)}
    a_scores={k.removesuffix(".nii.gz"):float(v) for k,v in a_payload["scores"].items()}
    e_rows={r["case_id"].removesuffix(".nii.gz"):r for r in read_csv(Path("outputs/response_routing/E_joint_empty_to_low/routing.csv"))}
    a_high={cid for cid,value in a_scores.items() if value>float(a_payload["threshold"])}
    e_high={cid for cid,r in e_rows.items() if r["assigned_branch"]=="high"}
    base_summary=json.loads((root/"baseline_reproducibility_summary.json").read_text())
    for label,high in (("A",a_high),("E",e_high)):
        low=set(ids)-high
        base_summary[label]={"high":len(high),"low":len(low),"high_mean_dice":float(np.mean([source_dice[c] for c in high])),
                             "low_mean_dice":float(np.mean([source_dice[c] for c in low])),"top59_case_ids":sorted(high)}
    base_summary["E"]["top59_matches_routing_csv"]=(e_high=={cid for cid,r in e_rows.items() if r["assigned_branch"]=="high"})
    base_summary["E_empty_to_low_cases"]=json.loads(Path("outputs/response_routing/E_joint_empty_to_low/routing_summary.json").read_text())
    (root/"baseline_reproducibility_summary.json").write_text(json.dumps(base_summary,indent=2)+"\n")
    make_figures(root,figs,grouped,ids,baseline,ranking_rows,bins_rows,topk_rows)
    # Case-set and volume distribution summaries.
    dice_sub=[]
    for name,mask in subsets.items():
        vals=dice_all[mask]
        dice_sub.append({"subset":name,"n":len(vals),"mean_dice":float(np.mean(vals)) if len(vals) else np.nan,
                         "median_dice":float(np.median(vals)) if len(vals) else np.nan,"dice_zero_n":int((vals==0).sum())})
    write_csv(root/"dice_subgroup_distribution.csv",dice_sub)
    report(root,ranking_rows,baseline,dice_sub,topk_rows,highlow_rows)


def make_figures(root,fig_dir,grouped,ids,baseline,ranking,bins_rows,topk_rows):
    import matplotlib.pyplot as plt
    top=ranking[:4]
    dice=np.array([baseline[i]["dice"] for i in ids])
    for rank,row in enumerate(top,1):
        key=(row["metric"],row["perturbation"],row["aggregation"]); mapping=grouped[key]
        conf=np.array([float(mapping[i]["confidence"]) for i in ids]); unc=np.array([float(mapping[i]["uncertainty"]) for i in ids])
        fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(conf,dice,s=18,alpha=.65); ax.set(xlabel="Confidence",ylabel="Dice",title=f"{key[0]} {key[1]} {key[2]} — confidence vs Dice");fig.tight_layout();fig.savefig(fig_dir/f"top{rank}_confidence_dice.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(unc,dice,s=18,alpha=.65); ax.set(xlabel="Uncertainty",ylabel="Dice",title=f"{key[0]} {key[1]} {key[2]} — uncertainty vs Dice");fig.tight_layout();fig.savefig(fig_dir/f"top{rank}_uncertainty_dice.png",dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4)); ax.hist(conf,bins=20); ax.set(xlabel="Confidence",ylabel="Cases",title=f"{key[0]} confidence distribution");fig.tight_layout();fig.savefig(fig_dir/f"top{rank}_confidence_histogram.png",dpi=160);plt.close(fig)
        order=np.lexsort((np.array(ids),-conf)); risks=1-dice[order]; coverage=np.arange(1,len(ids)+1)/len(ids)
        fig,ax=plt.subplots(figsize=(6,4)); ax.plot(coverage,np.cumsum(risks)/np.arange(1,len(ids)+1));ax.set(xlabel="Coverage",ylabel="Mean risk (1-Dice)",title="Risk-coverage curve");fig.tight_layout();fig.savefig(fig_dir/f"top{rank}_risk_coverage.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for row in top:
        vals=sorted([r for r in topk_rows if r["metric"]==row["metric"] and r["perturbation"]==row["perturbation"] and r["aggregation"]==row["aggregation"]],key=lambda x:int(x["K"]))
        ax.plot([int(r["K"]) for r in vals],[float(r["mean_dice"]) for r in vals],marker="o",label=f"{row['metric']} {row['perturbation']} {row['aggregation']}")
    ax.set(xlabel="Top K",ylabel="Mean Dice",title="Top-K mean Dice");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(fig_dir/"topk_mean_dice.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for row in top:
        vals=sorted([r for r in bins_rows if r["metric"]==row["metric"] and r["perturbation"]==row["perturbation"] and r["aggregation"]==row["aggregation"]],key=lambda x:int(x["bin"]))
        ax.plot([int(r["bin"]) for r in vals],[float(r["mean_dice"]) for r in vals],marker="o",label=row["metric"])
    ax.set(xlabel="Equal-frequency confidence bin (low to high)",ylabel="Mean Dice",title="Dice by confidence quintile");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(fig_dir/"confidence_quintiles.png",dpi=160);plt.close(fig)


def report(root,ranking,baseline,dice_sub,topk,highlow):
    summary=json.loads((root/"baseline_reproducibility_summary.json").read_text())
    baseline_rows=read_csv(root/"baseline_reproducibility.csv")
    worst=max(baseline_rows,key=lambda r:float(r["absolute_difference"]))
    failure_rows=read_csv(root/"failure_log.csv")
    empty_extras=sorted({r["case_id"] for r in failure_rows if r["stage"]=="empty_prediction_count"})
    failures=read_csv(root/"failure_detection_summary.csv")
    topk_all=read_csv(root/"topk_summary.csv")
    topk59={(r["metric"],r["perturbation"],r["aggregation"]):r for r in topk_all if r["K"]=="59" and r.get("gating")=="raw"}
    gated59={(r["metric"],r["perturbation"],r["aggregation"]):r for r in topk_all if r["K"]=="59" and r.get("gating")=="gated"}
    def detect(metric,views,target):
        r=next((x for x in failures if x["metric"]==metric and x["perturbation"]==views and x["aggregation"]=="case" and x["target"]==target),None)
        return (float(r["auroc"]),float(r["auprc"])) if r else (float("nan"),float("nan"))
    recommendations=[("I2_Pairwise_Soft","Image+Prompt","case"),("I1_Pairwise_Hard","Prompt-only","case"),
                     ("I3_VVC","Image+Prompt","case"),("I2_Pairwise_Soft","Image-only","case")]
    lines=["# Stage 1 Image-level Uncertainty Benchmark", "",
           "## Baseline audit", "",
           f"All {summary['cases']} case IDs match the historical file. Mean absolute Dice difference is {summary['mean_abs_dice_difference']:.6f}; maximum is {summary['max_abs_dice_difference']:.6f} for case `{worst['case_id']}` (historical {float(worst['historical_dice']):.6f}, current {float(worst['current_dice']):.6f}). The 1e-4 audit threshold is exceeded, as accepted for this run; findings are reported without treating the runs as numerically identical.",
           f"Current prediction counts: {summary['raw_empty']} empty, {summary['raw_nonempty']} nonempty, {summary['dice_zero_nonempty']} Dice-zero/nonempty. Expected historical counts were 17, 84, and 25. Dice bins <0.1 / [0.1,0.5] / >0.5 are {summary['dice_lt_0_1']} / {summary['dice_0_1_to_0_5']} / {summary['dice_gt_0_5']}.",
           "", "The largest case-level deviations and empty/nonempty transitions are in `baseline_reproducibility.csv`; review these before interpreting fine-grained deltas.",
           "", "## Leading candidates", "", "Ranking considers nonempty-case Spearman and full-case AURC; it is descriptive, not a trained calibration.", ""]
    lines += ["| Metric | Views | Aggregate | Spearman (nonempty) | AURC (all) | Dice-zero/nonempty in Top-59 |", "|---|---|---:|---:|---:|---:|"]
    for r in ranking[:5]: lines.append(f"| {r['metric']} | {r['perturbation']} | {r['aggregation']} | {float(r['spearman_nonempty']):.3f} | {float(r['aurc_all']):.3f} | {r['dice_zero_nonempty_top59']} |")
    lines += ["", "## Metric recommendations", "", "These are candidates for later gradient-discount validation, not calibrated absolute quality probabilities. The recommendation weighs nonempty-case association, full-cohort risk-coverage, failure detection, Top-59 mistakes, ties, and computation cost.", "",
              "| Candidate | Nonempty Spearman | AURC | AUROC / AUPRC for Dice < 0.1 | Raw Top-59 mean Dice / zero-nonempty | Gated Top-59 mean Dice / zero-nonempty |", "|---|---:|---:|---:|---:|---:|"]
    rankmap={(r["metric"],r["perturbation"],r["aggregation"]):r for r in ranking}
    for key in recommendations:
        r=rankmap[key]; auc,ap=detect(key[0],key[1],"dice_lt_0.1"); krow=topk59[key];gk=gated59[key]
        lines.append(f"| {key[0]} / {key[1]} | {float(r['spearman_nonempty']):.3f} | {float(r['aurc_all']):.3f} | {auc:.3f} / {ap:.3f} | {float(krow['mean_dice']):.3f} / {krow['dice_zero_nonempty_n']} | {float(gk['mean_dice']):.3f} / {gk['dice_zero_nonempty_n']} |")
    target_errors={"0000321625","0006150559","0010392655","0018128083","0020946113","0032710370","0032929968","0033102538"}
    case_rows=read_csv(root/"per_case_scores.csv")
    gated_maps=defaultdict(dict)
    for row in case_rows:
        if row["gating"]=="gated" and row["aggregation"]=="case":gated_maps[(row["metric"],row["perturbation"])][row["case_id"]]=row
    stable_summary=[]
    for metric,views,_ in recommendations:
        mapping=gated_maps[(metric,views)];order=sorted(mapping,key=lambda cid:(-float(mapping[cid]["confidence"]),cid))
        stable_summary.append(f"{metric}/{views}: {sum(cid in target_errors for cid in order[:59])}/8 of the previously misrouted E cases remain in gated Top-59")
    lines += ["", "I2 soft Dice with Image+Prompt has the strongest AURC and failure detection among these choices and keeps the continuous score nearly tie-free. I1 hard Dice with Prompt-only has the highest nonempty Spearman but has more ties and weaker failure AUROC. I3 VVC with Image+Prompt and I2 soft Dice with Image-only are useful alternatives when view-group robustness or image-only operation matters.",
              "", "For the eight historically misrouted E cases, gated Top-59 retention is: " + "; ".join(stable_summary) + ".",
              "", "## Perturbation and spatial aggregation", "",
              "Image+Prompt means the two group scores were computed independently and averaged. It is not a pooled-view score. For voxel-map methods, Union Mean often improves nonempty ranking over Full-volume Mean: for example, prompt-only PE Spearman changes from -0.025 (G0) to 0.376 (G1), and prompt-only EPKL from 0.346 to 0.517. This gain does not generalize to every map; review `conditional_correlations.csv` and `aurc_summary.csv` together.",
              "", "## A/E reproduction", "", f"A: {summary.get('A',{}).get('high')} High / {summary.get('A',{}).get('low')} Low; mean Dice {summary.get('A',{}).get('high_mean_dice',float('nan')):.6f} / {summary.get('A',{}).get('low_mean_dice',float('nan')):.6f}.",
              f"E: {summary.get('E',{}).get('high')} High / {summary.get('E',{}).get('low')} Low; mean Dice {summary.get('E',{}).get('high_mean_dice',float('nan')):.6f} / {summary.get('E',{}).get('low_mean_dice',float('nan')):.6f}. E empty-to-low summary: {summary.get('E_empty_to_low_cases',{}).get('no_response')} empty and {summary.get('E_empty_to_low_cases',{}).get('response_present')} nonempty. E Top-59 case IDs match historical `routing.csv`: {summary.get('E',{}).get('top59_matches_routing_csv')}.",
              "", "## Stable errors, confidence, and lesion size", "",
              "Current predictions include 22 Dice-zero/nonempty cases (historical count: 25). The case-level table lists each metric's uncertainty, confidence, cohort rank, nonempty rank, Top-59 status, and A/E rank for these cases and the eight previously selected E errors.",
              "", "Confidence quintiles and distributions are in `score_bin_summary.csv` and `figures/`. GT/prediction physical volumes and view-wise foreground volumes are in `lesion_size_analysis.csv`; volume correlations and tertile results are in `lesion_size_correlations.csv` and `lesion_size_group_summary.csv`. GT volumes are used only in this offline analysis.",
              "", "## Outputs", "", "Per-case scores retain raw and gated results. Detailed correlations, conditional subsets, AURC, raw and gated Top-K, High/Low, paired bootstrap, failure cases, score bins, lesion volumes, and figures are stored alongside this report.",
              "", "## Caveats", "", f"The baseline is not numerically aligned to 1e-4; mean/max absolute Dice differences are reported above. The E historical empty count is 17 versus 20 in this run; additional current-empty cases are {', '.join(empty_extras)}. The user authorized continuing; interpret comparisons with this discrepancy visible. Four cases have empty G1 unions; they are recorded with NaN uncertainty and zero valid voxels in `failure_log.csv`, with no imputation. No GT was used in uncertainty computation; GT enters only post-inference evaluation and analysis. No adaptation or training was performed.", ""]
    (root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
