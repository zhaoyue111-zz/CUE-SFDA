#!/usr/bin/env python3
"""Offline voxel-level uncertainty analysis using saved multiview probabilities."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero

from cue_sfda.benchmark.metrics import EPS, entropy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1 = ROOT / "analysis/hierarchical_uncertainty/stage1_image_level"
DEFAULT_OUT = ROOT / "analysis/hierarchical_uncertainty/stage2_voxel_level"
THRESHOLD = 0.5
HIST_BINS = 16384

METRIC_LABELS = {
    "P0_Chebyshev": "Chebyshev confidence uncertainty",
    "P1_MPC": "Maximum probability confidence uncertainty",
    "P2_Entropy": "Entropy",
    "P3_Predictive_Entropy": "Predictive entropy",
    "P4_Expected_Entropy": "Expected entropy",
    "P5_Predictive_Variance": "Predictive variance",
    "P6_Mutual_Information": "Mutual information",
    "P7_EPKL": "Expected pairwise KL divergence",
    "P8_Vote_Disagreement": "Vote disagreement",
    "P9_Image_Prompt_Disagreement": "Image–Prompt disagreement",
}
RANGES = {
    "P0_Chebyshev": (0.0, 1.0), "P1_MPC": (0.0, 0.5), "P2_Entropy": (0.0, 1.0),
    "P3_Predictive_Entropy": (0.0, 1.0), "P4_Expected_Entropy": (0.0, 1.0),
    "P5_Predictive_Variance": (0.0, 0.25), "P6_Mutual_Information": (0.0, 1.0),
    "P7_EPKL": (0.0, float(np.log((1.0-1e-6)/1e-6))), "P8_Vote_Disagreement": (0.0, 0.5),
    "P9_Image_Prompt_Disagreement": (0.0, 1.0),
}
REGIONS = ["Full_valid", "Prediction_foreground", "Multi_view_union"]
ERROR_TYPES = ["TP", "TN", "FP", "FN"]
TOP_PERCENTAGES = [1, 5, 10, 20]


def case_id(value):
    return str(value).removesuffix(".nii.gz").zfill(10)


def dice_bin(value):
    if not np.isfinite(value): return "undefined"
    if value < .1: return "Dice<0.1"
    if value <= .5: return "0.1<=Dice<=0.5"
    if value <= .8: return "0.5<Dice<=0.8"
    return "Dice>0.8"


def sliced(array, slicer):
    return array[(..., *slicer)]


def _stream_group_maps(views, device=None):
    """Compute Stage-1 map formulas with streaming reductions to limit memory."""
    if device is not None:
        # float64 is needed for the same 1e-8 entropy clipping used by Stage 1;
        # in float32, 1-1e-8 rounds to 1 and creates 0*log(0) NaNs.
        t=torch.as_tensor(views,device=device,dtype=torch.float64)
        k=int(t.shape[0]); pmean=t.mean(dim=0)
        variance=((t-pmean.unsqueeze(0))**2).mean(dim=0)
        pe=t.clamp(EPS,1-EPS)
        ent=-(pe*torch.log2(pe)+(1-pe)*torch.log2(1-pe))
        mean_ent=ent.mean(dim=0)
        pm=pmean.clamp(EPS,1-EPS)
        pred_ent=-(pm*torch.log2(pm)+(1-pm)*torch.log2(1-pm))
        pkl=t.clamp(1e-6,1-1e-6)
        pkmean=pkl.mean(dim=0)
        epkl=(pkl*torch.log(pkl)+(1-pkl)*torch.log1p(-pkl)).mean(dim=0) - pkmean*torch.log(pkl).mean(dim=0) - (1-pkmean)*torch.log1p(-pkl).mean(dim=0)
        votes=(t>THRESHOLD).sum(dim=0).to(torch.float32)/k
        maps={
            "P0_Chebyshev":variance/(variance+(pmean-THRESHOLD)**2+EPS),
            "P3_Predictive_Entropy":pred_ent,"P4_Expected_Entropy":mean_ent,
            "P5_Predictive_Variance":variance,"P6_Mutual_Information":torch.clamp(pred_ent-mean_ent,min=0),
            "P7_EPKL":torch.clamp(epkl,min=0),"P8_Vote_Disagreement":1-torch.maximum(votes,1-votes),
        }
        return {k:v.detach().cpu().numpy().astype(np.float32,copy=False) for k,v in maps.items()}
    k = int(views.shape[0])
    raw = views.astype(np.float64,copy=False)
    pmean = raw.mean(axis=0)
    variance = np.zeros(pmean.shape, dtype=np.float64)
    mean_ent = np.zeros(pmean.shape, dtype=np.float64)
    mean_logp = np.zeros(pmean.shape, dtype=np.float64)
    mean_logq = np.zeros(pmean.shape, dtype=np.float64)
    mean_base = np.zeros(pmean.shape, dtype=np.float64)
    pkl_mean = np.zeros(pmean.shape, dtype=np.float64)
    votes = np.zeros(pmean.shape, dtype=np.uint16)
    for p32 in views:
        p = np.clip(p32.astype(np.float64, copy=False), 1e-6, 1-1e-6)
        variance += (p32.astype(np.float64, copy=False) - pmean) ** 2 / k
        mean_ent += entropy(p32) / k
        mean_logp += np.log(p) / k
        mean_logq += np.log1p(-p) / k
        mean_base += (p*np.log(p)+(1-p)*np.log1p(-p)) / k
        pkl_mean += p/k
        votes += (p32 > THRESHOLD)
    pred_ent = entropy(pmean)
    epkl = mean_base - pkl_mean*mean_logp - (1-pkl_mean)*mean_logq
    maps = {
        "P0_Chebyshev": variance / (variance + (pmean-THRESHOLD)**2 + EPS),
        "P3_Predictive_Entropy": pred_ent,
        "P4_Expected_Entropy": mean_ent,
        "P5_Predictive_Variance": variance,
        "P6_Mutual_Information": np.maximum(pred_ent-mean_ent, 0.0),
        "P7_EPKL": np.maximum(epkl, 0.0),
        "P8_Vote_Disagreement": 1-np.maximum(votes/k, 1-votes/k),
    }
    return {k:np.asarray(v,dtype=np.float32) for k,v in maps.items()}


def build_maps(original, image_views, prompt_views, device=None):
    im = _stream_group_maps(image_views,device)
    pr = _stream_group_maps(prompt_views,device)
    maps = {}
    for metric in im:
        maps[(metric,"Image-only")] = im[metric]
        maps[(metric,"Prompt-only")] = pr[metric]
        maps[(metric,"Image+Prompt")] = .5*im[metric] + .5*pr[metric]
    if device is not None:
        o=torch.as_tensor(original,device=device,dtype=torch.float64)
        maps[("P1_MPC","Original")]=(1-torch.maximum(o,1-o)).cpu().numpy().astype(np.float32,copy=False)
        maps[("P2_Entropy","Original")]=(-(o*torch.log2(o.clamp(EPS,1-EPS))+(1-o)*torch.log2((1-o).clamp(EPS,1-EPS)))).cpu().numpy().astype(np.float32,copy=False)
        iv=torch.as_tensor(image_views,device=device,dtype=torch.float64)
        pv=torch.as_tensor(prompt_views,device=device,dtype=torch.float64)
        maps[("P9_Image_Prompt_Disagreement","Image+Prompt")]=torch.abs(iv.mean(0)-pv.mean(0)).cpu().numpy().astype(np.float32,copy=False)
        del o,iv,pv
    else:
        original = np.asarray(original,dtype=np.float32)
        maps[("P1_MPC","Original")] = 1-np.maximum(original,1-original)
        maps[("P2_Entropy","Original")] = entropy(original).astype(np.float32)
        maps[("P9_Image_Prompt_Disagreement","Image+Prompt")] = np.abs(
            image_views.mean(axis=0,dtype=np.float64)-prompt_views.mean(axis=0,dtype=np.float64)).astype(np.float32)
    return maps


def _hist_auc_ap(scores, labels, score_range):
    n = int(labels.size)
    positives = int(np.count_nonzero(labels))
    negatives = n-positives
    if n == 0:
        return np.nan,np.nan,"empty_region","empty_region"
    lo,hi=score_range
    scaled=np.floor((np.clip(scores,lo,hi)-lo)/(hi-lo)*HIST_BINS).astype(np.int32)
    scaled=np.clip(scaled,0,HIST_BINS-1)
    positive_hist=np.bincount(scaled[labels],minlength=HIST_BINS).astype(np.float64)
    total_hist=np.bincount(scaled,minlength=HIST_BINS).astype(np.float64)
    negative_hist=total_hist-positive_hist
    # Iterate bins from low to high for the Mann–Whitney form of AUROC.
    auc=np.nan
    auc_reason=""
    if positives and negatives:
        neg_below=np.cumsum(negative_hist)-negative_hist
        auc=float(np.sum(positive_hist*(neg_below+.5*negative_hist))/(positives*negatives))
    else:
        auc_reason="no_negative_voxels" if positives else "no_error_positive_voxels"
    # Average precision uses the descending score thresholds, with ties kept in one bin.
    ap=np.nan
    ap_reason=""
    if positives:
        pos_desc=positive_hist[::-1]; total_desc=total_hist[::-1]
        tp=np.cumsum(pos_desc); seen=np.cumsum(total_desc)
        ap=float(np.sum((pos_desc/positives)*(tp/np.maximum(seen,1))))
    else:
        ap_reason="no_error_positive_voxels"
    return auc,ap,auc_reason,ap_reason


def _top_mask(scores, percent):
    n=scores.size
    k=max(1,int(np.ceil(n*percent/100))) if n else 0
    if k==0:return np.zeros(scores.shape,dtype=bool)
    flat=scores.ravel()
    threshold=np.partition(flat,n-k)[n-k]
    above=flat>threshold
    need=k-int(above.sum())
    ties=np.flatnonzero(flat==threshold)
    chosen=np.zeros(n,dtype=bool)
    chosen[above]=True
    chosen[ties[:need]]=True
    return chosen.reshape(scores.shape)


def _approx_quantiles(values, score_range, probs=(.5,.9,.95), bins=1024):
    lo,hi=score_range
    hist,edges=np.histogram(values,bins=bins,range=(lo,hi))
    cumulative=np.cumsum(hist)
    answer=[]
    for q in probs:
        target=q*len(values)
        index=min(int(np.searchsorted(cumulative,target,side="left")),bins-1)
        previous=int(cumulative[index-1]) if index else 0
        count=max(int(hist[index]),1)
        fraction=np.clip((target-previous)/count,0,1)
        answer.append(float(edges[index]+fraction*(edges[index+1]-edges[index])))
    return answer


def metric_variant_order():
    order=[]
    for metric in ("P0_Chebyshev","P1_MPC","P2_Entropy","P3_Predictive_Entropy","P4_Expected_Entropy",
                   "P5_Predictive_Variance","P6_Mutual_Information","P7_EPKL","P8_Vote_Disagreement",
                   "P9_Image_Prompt_Disagreement"):
        variants=["Original"] if metric in ("P1_MPC","P2_Entropy") else (["Image+Prompt"] if metric=="P9_Image_Prompt_Disagreement" else ["Image-only","Prompt-only","Image+Prompt"])
        order.extend((metric,v) for v in variants)
    return order


def sample_cases(stage1, output):
    base=pd.read_csv(stage1/"baseline_reproducibility.csv",dtype={"case_id":str})
    base["case_id"]=base.case_id.map(case_id)
    e=pd.read_csv(ROOT/"outputs/response_routing/E_joint_empty_to_low/routing.csv",dtype={"case_id":str})
    e["case_id"]=e.case_id.map(case_id)
    ebranch=dict(zip(e.case_id,e.assigned_branch))
    estate=dict(zip(e.case_id,e.response_state))
    base=base.sort_values("case_id").reset_index(drop=True)
    selected=[]
    current_nonempty_zero=base[(base.original_foreground_voxels>0)&(base.current_dice==0)]
    group1=current_nonempty_zero[current_nonempty_zero.case_id.isin([c for c,b in ebranch.items() if str(b).lower()=="high"])]
    assert len(group1)==8, f"Expected 8 current E-high zero-Dice nonempty cases, found {len(group1)}"
    for _,r in group1.iterrows():selected.append(("E_high_zero_Dice_nonempty",r))
    group_specs=[
        ("empty_prediction_Dice_zero",base[(base.original_foreground_voxels==0)&(base.current_dice==0)]),
        ("Dice_0.1_to_0.5",base[(base.current_dice>=.1)&(base.current_dice<=.5)]),
        ("Dice_0.5_to_0.8",base[(base.current_dice>.5)&(base.current_dice<=.8)]),
        ("Dice_above_0.8",base[base.current_dice>.8]),
    ]
    seen={r.case_id for _,r in selected}
    for group,df in group_specs:
        available=df[~df.case_id.isin(seen)]
        if len(available)<4:raise ValueError(f"Only {len(available)} eligible current cases for {group}")
        for _,r in available.head(4).iterrows():
            selected.append((group,r));seen.add(r.case_id)
    assert len(selected)==24 and len({r.case_id for _,r in selected})==24
    rows=[]
    for category,r in selected:
        historical_response=estate.get(r.case_id,"unknown")
        historical_empty=str(historical_response).lower()=="no-response"
        current_empty=bool(r.original_foreground_voxels==0)
        rows.append({"case_id":r.case_id,"category":category,"historical_E_branch":ebranch.get(r.case_id,"unknown"),
                     "historical_E_response_state":historical_response,
                     "historical_prediction_status":"empty" if historical_empty else "nonempty",
                     "current_prediction_status":"empty" if current_empty else "nonempty",
                     "prediction_status_changed":historical_empty!=current_empty,
                     "historical_dice":r.historical_dice,"historical_dice_bin":dice_bin(r.historical_dice),
                     "current_dice":r.current_dice,"current_dice_bin":dice_bin(r.current_dice),
                     "dice_bin_changed":dice_bin(r.historical_dice)!=dice_bin(r.current_dice),
                     "original_foreground_voxels":int(r.original_foreground_voxels),
                     "prediction_empty":bool(r.original_foreground_voxels==0),
                     "historical_current_dice_abs_diff":r.absolute_difference})
    output.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows)
    result.to_csv(output/"selected_cases.csv",index=False)
    return result


def select_slice(gt,pred):
    scores=[]
    for z in range(gt.shape[0]):
        tp=np.count_nonzero(gt[z]&pred[z]);fp=np.count_nonzero(~gt[z]&pred[z]);fn=np.count_nonzero(gt[z]&~pred[z])
        scores.append(3*(fp+fn)+tp)
    return int(np.argmax(scores)) if max(scores,default=0)>0 else gt.shape[0]//2


def nifti_write(path, array_zyx, affine, dtype=None):
    arr=np.asarray(array_zyx,dtype=dtype) if dtype is not None else np.asarray(array_zyx)
    path.parent.mkdir(parents=True,exist_ok=True)
    nib.save(nib.Nifti1Image(arr.transpose(2,1,0),affine),str(path))


def plot_case(case_id_value, category, mri, gt, pred, maps, z, display_range, path):
    variants=metric_variant_order()
    fig,axes=plt.subplots(4,7,figsize=(22,13),dpi=130)
    flat=axes.ravel()
    for ax in flat:ax.axis("off")
    base=mri[z]
    vmin,vmax=display_range
    flat[0].imshow(base,cmap="gray",vmin=vmin,vmax=vmax);flat[0].set_title("Original MRI")
    flat[1].imshow(base,cmap="gray",vmin=vmin,vmax=vmax)
    if gt[z].any():flat[1].contour(gt[z],levels=[.5],colors=["#ffff00"],linewidths=.8)
    if pred[z].any():flat[1].contour(pred[z],levels=[.5],colors=["#00ffff"],linewidths=.8)
    flat[1].set_title("GT outline (yellow) / Pred (cyan)")
    tp=gt[z]&pred[z];fp=(~gt[z])&pred[z];fn=gt[z]&(~pred[z])
    overlay=np.zeros((*base.shape,3),dtype=np.float32)
    overlay[tp]=[.0,.85,.25];overlay[fp]=[1.,.05,.05];overlay[fn]=[.05,.3,1.]
    flat[2].imshow(base,cmap="gray",vmin=vmin,vmax=vmax);flat[2].imshow(overlay,alpha=.52)
    flat[2].set_title("TP green / FP red / FN blue")
    for i,(metric,variant) in enumerate(variants,start=3):
        ax=flat[i];ax.axis("on")
        img=ax.imshow(base,cmap="gray",vmin=vmin,vmax=vmax)
        lo,hi=RANGES[metric]
        h=ax.imshow(np.clip(maps[(metric,variant)][z],lo,hi),cmap="magma",vmin=lo,vmax=hi,alpha=.48)
        ax.set_title(f"{metric}\n{variant}",fontsize=7)
        ax.set_xticks([]);ax.set_yticks([])
        cb=fig.colorbar(h,ax=ax,fraction=.046,pad=.02,shrink=.78)
        cb.ax.set_ylabel("Uncertainty",rotation=270,labelpad=4,fontsize=5)
    fig.suptitle(f"{case_id_value} | {category} | slice z={z} | MRI window {vmin:.3g}–{vmax:.3g}",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97))
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,bbox_inches="tight");plt.close(fig)


def finalize_results(out, stage1, selected, display_range, per_case, regions_df, coverage_df, slice_rows):
    for frame in (per_case,regions_df,coverage_df):
        frame["case_id"]=frame.case_id.map(case_id)
    per_case.to_csv(out/"per_case_metrics.csv",index=False)
    regions_df.to_csv(out/"region_uncertainty_statistics.csv",index=False)
    coverage_df.to_csv(out/"error_coverage_summary.csv",index=False)
    pd.DataFrame(slice_rows).to_csv(out/"visualization_slices.csv",index=False)
    group_keys=["metric","perturbation","region"]
    summary=[]
    for keys,grp in per_case.groupby(group_keys,dropna=False):
        row=dict(zip(group_keys,keys));row["cases"]=int(grp.case_id.nunique())
        for col in ["voxel_AUROC","voxel_AUPRC","error_fraction","mean_uncertainty"]:
            x=grp[col].dropna().to_numpy(float)
            row[f"{col}_case_mean"]=float(x.mean()) if x.size else np.nan
            row[f"{col}_case_median"]=float(np.median(x)) if x.size else np.nan
            row[f"{col}_case_std"]=float(x.std()) if x.size else np.nan
            row[f"{col}_valid_cases"]=int(x.size)
        row["AUROC_undefined_cases"]=int(grp.voxel_AUROC.isna().sum())
        row["AUPRC_undefined_cases"]=int(grp.voxel_AUPRC.isna().sum())
        for p in TOP_PERCENTAGES:
            for error_name in ("FP","FN"):
                col=f"top{p}_coverage_{error_name}"
                x=grp[col].dropna().to_numpy(float)
                row[f"{col}_case_mean"]=float(x.mean()) if x.size else np.nan
                row[f"{col}_case_valid_cases"]=int(x.size)
        for err_name in ERROR_TYPES:
            g=regions_df[(regions_df.metric==keys[0])&(regions_df.perturbation==keys[1])&
                         (regions_df.region==keys[2])&(regions_df.error_type==err_name)]
            x=g.mean_uncertainty.dropna().to_numpy(float)
            row[f"{err_name}_mean_uncertainty_case_mean"]=float(x.mean()) if x.size else np.nan
            row[f"{err_name}_cases_with_voxels"]=int(x.size)
        summary.append(row)
    summary_df=pd.DataFrame(summary)
    summary_df.to_csv(out/"metric_summary.csv",index=False)
    selected.to_csv(out/"selected_cases.csv",index=False)
    pd.DataFrame(slice_rows).to_csv(out/"visualization_slices.csv",index=False)
    baseline=pd.read_csv(stage1/"baseline_reproducibility.csv",dtype={"case_id":str})
    baseline["case_id"]=baseline.case_id.map(case_id)
    zero_nonempty=set(baseline.loc[(baseline.current_dice==0)&(baseline.original_foreground_voxels>0),"case_id"])
    empty_with_gt=set(baseline.loc[(baseline.original_foreground_voxels==0)&(baseline.gt_foreground_voxels>0),"case_id"])
    special=[]
    for subgroup,cids in [("Dice_zero_prediction_nonempty",zero_nonempty),("empty_prediction_GT_lesion",empty_with_gt)]:
        for (metric,variant,region),grp in per_case[per_case.case_id.isin(cids)].groupby(["metric","perturbation","region"]):
            ids=set(grp.case_id)
            reg=regions_df[(regions_df.case_id.isin(ids))&(regions_df.metric==metric)&
                           (regions_df.perturbation==variant)&(regions_df.region==region)]
            row={"subgroup":subgroup,"metric":metric,"perturbation":variant,"region":region,
                 "subgroup_cases":len(cids),"cases_present":grp.case_id.nunique(),
                 "voxel_AUROC_case_mean":float(grp.voxel_AUROC.mean()),
                 "voxel_AUPRC_case_mean":float(grp.voxel_AUPRC.mean()),
                 "FN_mean_uncertainty_case_mean":float(reg[reg.error_type=="FN"].mean_uncertainty.mean()),
                 "FN_cases_with_voxels":int(reg[reg.error_type=="FN"].mean_uncertainty.notna().sum()),
                 "FP_mean_uncertainty_case_mean":float(reg[reg.error_type=="FP"].mean_uncertainty.mean()),
                 "FP_cases_with_voxels":int(reg[reg.error_type=="FP"].mean_uncertainty.notna().sum())}
            for pct in TOP_PERCENTAGES:
                for error_name in ("FP","FN"):
                    col=f"top{pct}_coverage_{error_name}"
                    values=grp[col].dropna().to_numpy(float)
                    row[f"{col}_case_mean"]=float(values.mean()) if values.size else np.nan
                    row[f"{col}_valid_cases"]=int(values.size)
            special.append(row)
    pd.DataFrame(special).to_csv(out/"special_case_response.csv",index=False)
    table=[]
    for metric in METRIC_LABELS:
        subset=summary_df[summary_df.metric==metric]
        if subset.empty:continue
        variant="Original" if metric in ("P1_MPC","P2_Entropy") else "Image+Prompt"
        full=subset[(subset.perturbation==variant)&(subset.region=="Full_valid")]
        if len(full):
            r=full.iloc[0]
            table.append(f"| {metric} | {r.voxel_AUROC_case_mean:.4f} | {r.voxel_AUPRC_case_mean:.4f} | {r.FP_mean_uncertainty_case_mean:.4g} | {r.FN_mean_uncertainty_case_mean:.4g} | {r.TP_mean_uncertainty_case_mean:.4g} | {r.TN_mean_uncertainty_case_mean:.4g} |")
    report=f'''# Stage 2 voxel-level uncertainty analysis

## Scope and protocol

All scores were computed offline from the 101 saved multiview probability caches; there was no model inference, training, parameter update, adaptation, or gradient discount. GT is used only to select/visualize cases and evaluate maps. Per-case macro statistics are reported so larger volumes do not dominate. The upstream image-level historical baseline audit remains unresolved (mean Dice difference about 6e-4 after checking the repository's original VoxTellPredictor); this voxel analysis is conditional on the current benchmark outputs.

The 24 cases are fixed in `selected_cases.csv`: 8 historical E-High and currently nonempty Dice-zero cases; 4 currently empty Dice-zero cases; 4 in Dice [0.1,0.5]; 4 in (0.5,0.8]; and 4 above 0.8. Case status uses the current benchmark. The selected cases have {int(selected.prediction_status_changed.sum())} historical/current prediction-empty status changes and {int(selected.dice_bin_changed.sum())} Dice-bin changes; both are recorded per case. All cases use the same selected slice across every map, listed in `visualization_slices.csv`.

P0 and P3–P8 were computed for Image-only and Prompt-only independently. Image+Prompt is the equal-weight voxel-map mean `0.5*Image + 0.5*Prompt`, preserving the requested modality balance. P1 MPC and P2 entropy use the original standard-prompt prediction and are marked `Original`; P9 is Image–Prompt mean-probability disagreement. Formulas follow `src/cue_sfda/benchmark/metrics.py`; no spatial aggregation is used. P7 is reported in nats. AUROC/AUPRC use a {HIST_BINS:,}-bin fixed-range histogram approximation for scalable per-case evaluation. The fixed mathematical display ranges used by visualizations are encoded in the analysis script; no slice-wise min–max normalization is applied.

The three evaluation regions are full valid inference ROI, original prediction foreground, and the union of original/image-view/prompt-view foreground. The binary error target is FP or FN. `per_case_metrics.csv` records case-level AUROC/AUPRC and FP/FN coverage at top 1%, 5%, 10%, and 20% uncertainty. When an evaluation region has no positives or negatives, the corresponding undefined score is NaN with a reason. `region_uncertainty_statistics.csv` gives TP/TN/FP/FN uncertainty distributions for each case, map, and region. `special_case_response.csv` separately summarizes the 22 zero-Dice nonempty predictions and empty-prediction cases with GT lesions, including lesion FN uncertainty in the full ROI and multiview union.

## Case-macro full valid ROI summary

| Metric | AUROC | AUPRC | FP mean U | FN mean U | TP mean U | TN mean U |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Read modality-specific comparisons, prediction-foreground/union analyses, top-uncertainty coverage, and per-case values from `metric_summary.csv`, `per_case_metrics.csv`, and `error_coverage_summary.csv`. These are descriptive; there is no automatically selected winner.

## Visualization and spatial files

Each selected case has a fixed-range overlay montage at `visualizations/<case>.png` with MRI, GT/Prediction contours, TP/FP/FN colors, and all applicable P0–P9 uncertainty overlays, including modality comparisons. MRI display window is common across selected cases (pooled ROI 1st–99th percentile: {display_range[0]:.6g} to {display_range[1]:.6g}); overlay alpha and magma color map are common. Each metric uses its fixed theoretical display range, never per-slice normalization.

`uncertainty_maps/<case>/` contains each selected case's complete 3D `.nii.gz` maps plus MRI reference, GT, prediction, and valid ROI layers. Files are written in the MRI reader's canonical reoriented geometry with the same physical affine; non-ROI voxels are zero. Load `MRI_reference.nii.gz` as the base in ITK-SNAP and add the maps/labels as overlays.

## Reading the results

The summaries compare error responses (FP/FN) with correct regions (TP/TN) for each perturbation, and show whether uncertainty concentrates on errors in the full ROI, foreground, and multiview union. Empty-prediction cases have no prediction-foreground region, so those region metrics are undefined; their GT lesion response is evaluated in the full ROI and multiview union. No single metric is declared best. Choose based on these macro statistics and the aligned visualizations.
'''
    (out/"REPORT.md").write_text(report)


def process_all(stage1, out, selected, image_dir, gt_dir, threshold=.5, worker_rank=None, world=1, device=None):
    cache_dir=stage1/"probabilities"
    files=sorted(cache_dir.glob("*.npz"))
    if len(files)!=101:raise RuntimeError(f"Expected complete 101-case probability cache, got {len(files)}")
    reader=NibabelIOWithReorient()
    if device is not None:
        torch.cuda.set_device(device)
        torch.cuda.init()
    selected_by_id={r.case_id:r for r in selected.itertuples(index=False)}
    case_metric_rows=[];region_rows=[];coverage_rows=[];slice_rows=[]
    # A common raw-MRI display window across all selected cases, estimated by reproducible spatial sampling.
    rng=np.random.default_rng(20261126);intensity_sample=[]
    for cid in selected_by_id:
        arr,_=reader.read_images([str(image_dir/f"{cid}.nii.gz")]);vol=arr[0].ravel()
        good=vol[np.isfinite(vol)&(vol!=0)]
        if good.size:intensity_sample.append(good[rng.choice(good.size,min(50000,good.size),replace=False)])
    display=np.concatenate(intensity_sample) if intensity_sample else np.array([0.,1.])
    display_range=tuple(float(x) for x in np.percentile(display,[1,99]))
    selected_map_root=out/"uncertainty_maps";visual_root=out/"visualizations"
    files=[p for i,p in enumerate(files) if worker_rank is None or i%world==worker_rank]

    for case_index,file in enumerate(files,1):
        cid=case_id(file.stem)
        with np.load(file,allow_pickle=False) as z:
            original=z["original"].astype(np.float32,copy=False)
            iv=z["image_views"].astype(np.float32,copy=False)
            pv=z["prompt_views"].astype(np.float32,copy=False)
            metadata=json.loads(str(z["metadata"].item()))
        if original.ndim==4 and original.shape[0]==1:original=original[0]
        gt_path=gt_dir/f"{cid}.nii.gz"
        if not gt_path.is_file():raise FileNotFoundError(gt_path)
        image,properties=reader.read_images([str(image_dir/f"{cid}.nii.gz")])
        gt_raw,_=reader.read_seg(str(gt_path))
        if gt_raw.ndim==4 and gt_raw.shape[0]==1:gt_raw=gt_raw[0]
        image_zyx=image[0].astype(np.float32,copy=False)
        # Cache roi_slicer is the crop inside the padded inference tensor. Restore
        # to the source MRI using the actual nonzero-crop bounding box instead.
        _,_,bbox=crop_to_nonzero(image,None)
        slicer=tuple(slice(int(pair[0]),int(pair[1])) for pair in bbox)
        roi_image=image_zyx[slicer]
        roi_gt=(gt_raw[slicer]>0)
        if roi_image.shape!=original.shape or roi_gt.shape!=original.shape:
            raise ValueError(f"ROI mismatch {cid}: image={roi_image.shape}, GT={roi_gt.shape}, cache={original.shape}")
        pred=original>threshold
        valid=np.ones(original.shape,dtype=bool)
        union=pred|(iv>threshold).any(axis=0)|(pv>threshold).any(axis=0)
        regions={"Full_valid":valid,"Prediction_foreground":pred,"Multi_view_union":union}
        tp=pred&roi_gt;tn=(~pred)&(~roi_gt);fp=pred&(~roi_gt);fn=(~pred)&roi_gt
        labels=fp|fn
        maps=build_maps(original,iv,pv,device)
        for metric,variant in metric_variant_order():
            umap=maps[(metric,variant)]
            lo,hi=RANGES[metric]
            umap=np.clip(umap,lo,hi)
            for region_name,region_mask in regions.items():
                scores=umap[region_mask].astype(np.float32,copy=False)
                error=labels[region_mask]
                auc,ap,auc_reason,ap_reason=_hist_auc_ap(scores,error,(lo,hi))
                row={"case_id":cid,"metric":metric,"perturbation":variant,"region":region_name,
                     "n_voxels":int(region_mask.sum()),"error_voxels":int(error.sum()),
                     "error_fraction":float(error.mean()) if error.size else np.nan,
                     "voxel_AUROC":auc,"voxel_AUPRC":ap,"AUROC_undefined_reason":auc_reason,
                     "AUPRC_undefined_reason":ap_reason,"mean_uncertainty":float(scores.mean()) if scores.size else np.nan,
                     "uncertainty_histogram_bins":HIST_BINS}
                top_masks={p:_top_mask(scores,p) for p in TOP_PERCENTAGES}
                for p in TOP_PERCENTAGES:
                    top=top_masks[p]
                    n_top=int(top.sum())
                    for error_name,error_type in [("FP",fp),("FN",fn)]:
                        denom=int((error_type&region_mask).sum())
                        num=int((top & (error_type[region_mask])).sum()) if n_top else 0
                        row[f"top{p}_coverage_{error_name}"]=float(num/denom) if denom else np.nan
                        row[f"top{p}_coverage_{error_name}_reason"]="" if denom else f"no_{error_name}_voxels_in_region"
                case_metric_rows.append(row)
                for err_name,err_mask in zip(ERROR_TYPES,(tp,tn,fp,fn)):
                    values=umap[err_mask&region_mask]
                    q50,q90,q95=_approx_quantiles(values,(lo,hi)) if values.size else (np.nan,np.nan,np.nan)
                    region_rows.append({"case_id":cid,"metric":metric,"perturbation":variant,"region":region_name,
                                        "error_type":err_name,"voxel_count":int(values.size),
                                        "mean_uncertainty":float(values.mean()) if values.size else np.nan,
                                        "std_uncertainty":float(values.std()) if values.size else np.nan,
                                        "median_uncertainty_hist_approx":q50,
                                        "p90_uncertainty_hist_approx":q90,"p95_uncertainty_hist_approx":q95,
                                        "quantile_histogram_bins":1024,
                                        "undefined_reason":"empty_confusion_region" if values.size==0 else ""})
                for p in TOP_PERCENTAGES:
                    top=top_masks[p]
                    for err_name,err_mask in [("FP",fp),("FN",fn)]:
                        denom=int((err_mask&region_mask).sum())
                        num=int(np.count_nonzero(top&err_mask[region_mask])) if top.size else 0
                        coverage_rows.append({"case_id":cid,"metric":metric,"perturbation":variant,"region":region_name,
                                             "top_percent":p,"error_type":err_name,"error_voxels":denom,
                                             "top_uncertainty_error_voxels":num,
                                             "coverage":float(num/denom) if denom else np.nan,
                                             "undefined_reason":"" if denom else f"no_{err_name}_voxels_in_region"})

        if cid in selected_by_id:
            record=selected_by_id[cid]
            zslice=select_slice(roi_gt,pred)
            slice_rows.append({"case_id":cid,"category":record.category,"slice_z_roi":zslice,
                               "GT_voxels_on_slice":int(roi_gt[zslice].sum()),"Pred_voxels_on_slice":int(pred[zslice].sum()),
                               "TP_on_slice":int(tp[zslice].sum()),"FP_on_slice":int(fp[zslice].sum()),"FN_on_slice":int(fn[zslice].sum())})
            case_map_dir=selected_map_root/cid;case_map_dir.mkdir(parents=True,exist_ok=True)
            affine=np.asarray(properties["nibabel_stuff"]["reoriented_affine"],dtype=float)
            nib.save(nib.Nifti1Image(image_zyx.transpose(2,1,0),affine),str(case_map_dir/"MRI_reference.nii.gz"))
            full_gt=np.zeros(image_zyx.shape,dtype=np.uint8);full_gt[slicer]=roi_gt.astype(np.uint8)
            full_pred=np.zeros(image_zyx.shape,dtype=np.uint8);full_pred[slicer]=pred.astype(np.uint8)
            full_valid=np.zeros(image_zyx.shape,dtype=np.uint8);full_valid[slicer]=1
            nifti_write(case_map_dir/"GT.nii.gz",full_gt,affine,np.uint8)
            nifti_write(case_map_dir/"Prediction.nii.gz",full_pred,affine,np.uint8)
            nifti_write(case_map_dir/"Valid_inference_ROI.nii.gz",full_valid,affine,np.uint8)
            for (metric,variant),umap in maps.items():
                full=np.zeros(image_zyx.shape,dtype=np.float32);full[slicer]=np.clip(umap,*RANGES[metric])
                nifti_write(case_map_dir/f"{metric}__{variant.replace('+','plus').replace(' ','_')}.nii.gz",full,affine,np.float32)
            plot_case(cid,record.category,roi_image,roi_gt,pred,maps,zslice,display_range,visual_root/f"{cid}.png")
            del affine,full_gt,full_pred,full_valid
        del maps,iv,pv,original,image,gt_raw
        if device is not None and case_index%10==0:
            print(f"[GPU {device}] processed {case_index}/{len(files)} assigned cases",flush=True)

    per_case=pd.DataFrame(case_metric_rows)
    regions_df=pd.DataFrame(region_rows)
    coverage_df=pd.DataFrame(coverage_rows)
    if worker_rank is not None:
        out.mkdir(parents=True,exist_ok=True)
        per_case.to_csv(out/"per_case_metrics.csv",index=False)
        regions_df.to_csv(out/"region_uncertainty_statistics.csv",index=False)
        coverage_df.to_csv(out/"error_coverage_summary.csv",index=False)
        pd.DataFrame(slice_rows).to_csv(out/"visualization_slices.csv",index=False)
        (out/"worker_run.json").write_text(json.dumps({"worker_rank":worker_rank,"world":world,"device":device,
                                                         "cases":len(files),"display_range":display_range,
                                                         "peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated(device)) if device is not None else 0},indent=2)+"\n")
        return
    finalize_results(out,stage1,selected,display_range,per_case,regions_df,coverage_df,slice_rows)
    print(f"Wrote {out}; {len(per_case)} per-case metric rows, {len(regions_df)} region rows, {len(coverage_df)} coverage rows")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage1",type=Path,default=DEFAULT_STAGE1)
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--image-dir",type=Path,default=Path("/data/hilab/datasets/WCH_MRI145/DWI/images"))
    ap.add_argument("--gt-dir",type=Path,default=Path("/data/hilab/datasets/WCH_MRI145/DWI/labels"))
    ap.add_argument("--devices",type=int,nargs="+",default=[0,1])
    ap.add_argument("--worker-rank",type=int)
    ap.add_argument("--world",type=int,default=1)
    ap.add_argument("--gpu-id",type=int)
    ap.add_argument("--selected-csv",type=Path)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.worker_rank is not None:
        selected=pd.read_csv(args.selected_csv,dtype={"case_id":str})
        process_all(args.stage1,args.out,selected,args.image_dir,args.gt_dir,worker_rank=args.worker_rank,
                    world=args.world,device=args.gpu_id)
        return
    selected=sample_cases(args.stage1,args.out)
    if len(args.devices)<2:raise ValueError("Official offline run requires both GPUs; pass --devices 0 1")
    worker_root=args.out/".workers"
    worker_root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    srcpath=str(ROOT/"src")
    env["PYTHONPATH"]=srcpath+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    procs=[]
    for rank,gpu in enumerate(args.devices):
        worker_out=worker_root/f"gpu{gpu}"
        command=[sys.executable,str(Path(__file__).resolve()),"--stage1",str(args.stage1.resolve()),
                 "--out",str(worker_out.resolve()),"--image-dir",str(args.image_dir),"--gt-dir",str(args.gt_dir),
                 "--devices",*map(str,args.devices),"--worker-rank",str(rank),"--world",str(len(args.devices)),
                 "--gpu-id",str(gpu),"--selected-csv",str((args.out/"selected_cases.csv").resolve())]
        procs.append((gpu,subprocess.Popen(command,env=env,cwd=ROOT)))
    print(f"Started {len(procs)} offline probability-map workers on GPUs {[g for g,_ in procs]}; no model is loaded.",flush=True)
    statuses=[(gpu,p.wait()) for gpu,p in procs]
    if any(code for _,code in statuses):raise RuntimeError(f"GPU workers failed: {statuses}")
    worker_dirs=[worker_root/f"gpu{gpu}" for gpu,_ in procs]
    per_case=pd.concat([pd.read_csv(w/"per_case_metrics.csv") for w in worker_dirs],ignore_index=True)
    region_stats=pd.concat([pd.read_csv(w/"region_uncertainty_statistics.csv") for w in worker_dirs],ignore_index=True)
    coverage=pd.concat([pd.read_csv(w/"error_coverage_summary.csv") for w in worker_dirs],ignore_index=True)
    slices=pd.concat([pd.read_csv(w/"visualization_slices.csv") for w in worker_dirs],ignore_index=True)
    for name in ("uncertainty_maps","visualizations"):(args.out/name).mkdir(parents=True,exist_ok=True)
    worker_meta=[]
    for w in worker_dirs:
        worker_meta.append(json.loads((w/"worker_run.json").read_text()))
        for case_dir in (w/"uncertainty_maps").glob("*"):
            dest=args.out/"uncertainty_maps"/case_dir.name
            if dest.exists():shutil.rmtree(dest)
            shutil.move(str(case_dir),str(dest))
        for fig in (w/"visualizations").glob("*.png"):
            shutil.move(str(fig),str(args.out/"visualizations"/fig.name))
    display_range=tuple(worker_meta[0]["display_range"])
    finalize_results(args.out,args.stage1,selected,display_range,per_case,region_stats,coverage,slices.to_dict("records"))
    (args.out/"gpu_run.json").write_text(json.dumps({"devices":[g for g,_ in procs],"worker_status":statuses,
                                                     "workers":worker_meta,"model_loaded":False,"inference_performed":False},indent=2)+"\n")
    shutil.rmtree(worker_root)
    print(f"Completed offline evaluation from {len(per_case.case_id.unique())} cases on GPUs {[g for g,_ in procs]}.",flush=True)


if __name__=="__main__":main()
