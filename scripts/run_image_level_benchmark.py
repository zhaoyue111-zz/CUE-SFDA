#!/usr/bin/env python3
"""Infer aligned multi-view probabilities and score the Stage 1 benchmark.

Two-device runs are launched as two independent, deterministic case shards. A
completed case is stored atomically, so interrupted runs can be resumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cue_sfda.data import TargetVolumeStore, load_split
from cue_sfda.method.augmentations import StrongViewGenerator
from cue_sfda.method.dynamic_hard import source_checkpoint_signature
from cue_sfda.modeling.lora import lora_disabled
from cue_sfda.modeling.voxtell import build_runtime, load_prompt_embedding, _prompt_cache_path
from cue_sfda.benchmark.metrics import (
    aggregate, confidence, entropy, hard_dice, pairwise_hard, pairwise_soft,
    uncertainty_maps, vvc,
)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def predict_group(runtime, store, volume, slicers, geometry, embedding, amp, views=None, view_count=1):
    device = runtime.device
    padded_shape = tuple(geometry["padded_shape"])
    roi_slicer = geometry["roi_slicer"]
    accumulation_dtype = torch.float16 if amp else torch.float32
    gaussian = compute_gaussian(runtime.patch_size, sigma_scale=1 / 8, value_scaling_factor=10,
                                dtype=accumulation_dtype, device=device)
    accumulations = [torch.zeros(padded_shape, dtype=accumulation_dtype, device=device) for _ in range(view_count)]
    normalization = torch.zeros(padded_shape, dtype=accumulation_dtype, device=device)
    with torch.inference_mode(), lora_disabled(runtime.model):
        for slicer in slicers:
            patch = store.extract_patch(volume, slicer).unsqueeze(0).to(device)
            patch_views = views.make_uncertainty_views(patch) if views is not None else None
            if isinstance(embedding, (tuple, list)):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = runtime.model(patch, torch.cat(list(embedding), dim=1))
                for k in range(view_count):
                    accumulations[k][slicer] += logits[0, k].to(accumulation_dtype) * gaussian
            else:
                for k in range(view_count):
                    image = patch if patch_views is None else patch_views[k].image
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        logits = runtime.model(image, embedding)
                    fused_value = logits[0, 0].to(accumulation_dtype)
                    if patch_views is not None:
                        fused_value = patch_views[k].align(fused_value)
                    accumulations[k][slicer] += fused_value * gaussian
            normalization[slicer] += gaussian
    outputs = []
    for accumulation in accumulations:
        outputs.append(torch.sigmoid(torch.div(accumulation, normalization.clamp_min(1e-6)))[roi_slicer].cpu().numpy().astype(np.float32))
    return np.stack(outputs)


def case_worker(args, cfg, output: Path, rank: int, world: int):
    device_id = args.devices[rank]
    cfg["experiment"]["gpu_id"] = int(device_id)
    torch.cuda.set_device(device_id)
    torch.manual_seed(int(cfg["experiment"]["seed"]) + rank)
    np.random.seed(int(cfg["experiment"]["seed"]) + rank)
    train_cases, _ = load_split(cfg["data"]["image_dir"], cfg["data"]["split_json"])
    cases = sorted(train_cases, key=lambda c: c.name)
    shard = [case for i, case in enumerate(cases) if i % world == rank]
    if args.limit_cases is not None:
        shard = shard[:max(0, (args.limit_cases - rank + world - 1) // world)] if rank < args.limit_cases else []
    logger = logging.getLogger(f"image_benchmark_gpu{device_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "inference_log.csv", encoding="utf-8")
    # Logs are written as CSV records manually to avoid concurrent handlers.
    handler.close()
    store = TargetVolumeStore(tuple(cfg["data"]["patch_size"]), cache_size=1)
    prompt_cache = output / "prompt_cache_fp16"
    prompt_cache.mkdir(parents=True, exist_ok=True)
    standard = str(cfg["model"]["prompt"])
    standard_source = ROOT / "outputs/response_routing/E_joint_empty_to_low/prompt_cache/liver_cancer_aef3cc625786e0ac.pt"
    standard_target = _prompt_cache_path(prompt_cache, standard, Path(cfg["model"]["text_encoder"]).expanduser().resolve())
    if not standard_target.exists() and standard_source.is_file():
        shutil.copy2(standard_source, standard_target)
    runtime = build_runtime(cfg, prompt_cache)
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    runtime.model.eval()
    prompt_names = [str(x) for x in cfg["method"]["routing_prompts"]]
    embeddings = {p: load_prompt_embedding(cfg, p, prompt_cache, runtime.device) for p in prompt_names}
    if cfg["model"]["prompt"] not in embeddings:
        embeddings[cfg["model"]["prompt"]] = runtime.embedding
    image_perturb = StrongViewGenerator(cfg["method"], int(cfg["experiment"]["seed"]) + 1)
    amp = bool(cfg["model"].get("amp", True)) and runtime.device.type == "cuda"
    for case in shard:
        case_id = case.name[:-7] if case.name.lower().endswith(".nii.gz") else Path(case.name).stem
        target = output / "probabilities" / f"{case_id}.npz"
        if target.is_file() and not args.force:
            try:
                with np.load(target, allow_pickle=False) as z:
                    if str(z["case_id"].item()) == case_id:
                        continue
            except Exception:
                pass
        volume, slicers, geometry = store.sliding_window_with_geometry(case, float(cfg["train"]["tile_step_size"]))
        original = predict_group(runtime, store, volume, slicers, geometry,
                                 embeddings[str(cfg["model"]["prompt"])], amp)
        image_views = predict_group(runtime, store, volume, slicers, geometry,
                                    embeddings[str(cfg["model"]["prompt"])], amp,
                                    image_perturb, int(cfg["method"]["num_views"]))
        prompt_views = np.concatenate([
            predict_group(runtime, store, volume, slicers, geometry, embeddings[p], amp)
            for p in prompt_names
        ], axis=0)
        metadata = {"case_id": case_id,
                    "roi_shape": list(geometry["roi_shape"]),
                    "padded_shape": list(geometry["padded_shape"]),
                    "roi_slicer": [[int(s.start or 0), int(s.stop)] for s in geometry["roi_slicer"]],
                    "spacing": list(geometry["spacing"]),
                    "patch_size": list(runtime.patch_size),
                    "tile_step_size": float(cfg["train"]["tile_step_size"]),
                    "image_views_include_original": False,
                    "prompt_views_include_original": str(cfg["model"]["prompt"]) in prompt_names,
                    "image_view_count": len(image_views), "prompt_view_count": len(prompt_views),
                    "prompts": prompt_names,
                    "model_checkpoint": source_checkpoint_signature(cfg["model"])}
        atomic_npz(target, case_id=np.asarray(metadata["case_id"]), original=original[0],
                   image_views=image_views, prompt_views=prompt_views,
                   metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        with (output / "inference_log.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([metadata["case_id"], device_id, len(slicers), "complete", target.stat().st_size])
        logger.info("complete %s patches=%d shape=%s", case.name, len(slicers), metadata["roi_shape"])


def score_all(args, cfg, output):
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    train_cases, _ = load_split(cfg["data"]["image_dir"], cfg["data"]["split_json"])
    case_names = sorted(c.name[:-7] if c.name.lower().endswith(".nii.gz") else Path(c.name).stem for c in train_cases)
    gt_dir = Path(args.gt_dir)
    threshold = float(cfg["method"]["probability_threshold"])
    reference_path = Path(args.reference_dir)
    hist_csv = reference_path / "analysis/tables/step2_routing_case_join.csv"
    hist = {r["case_id"].removesuffix(".nii.gz"): float(r["zero_shot_dice"]) for r in csv.DictReader(hist_csv.open())}
    a_path = reference_path / "outputs/dwi_only_liver_cancer/confidence_partition.json"
    a_payload = json.load(a_path.open())
    a_scores = {k.removesuffix(".nii.gz"): float(v) for k,v in a_payload["scores"].items()}
    e_route_path = ROOT / "outputs/response_routing/E_joint_empty_to_low/routing_summary.json"
    e_route = json.load(e_route_path.open())
    rows, baseline = [], []
    for case_id in case_names:
        pred_file = output / "probabilities" / f"{case_id}.npz"
        if not pred_file.is_file():
            continue
        with np.load(pred_file, allow_pickle=False) as z:
            original, iv, pv = z["original"], z["image_views"], z["prompt_views"]
            metadata = json.loads(str(z["metadata"].item()))
        if original.ndim == 4 and original.shape[0] == 1:
            original = original[0]
        gt_file = gt_dir / f"{case_id}.nii.gz"
        if not gt_file.is_file():
            raise FileNotFoundError(gt_file)
        reader = NibabelIOWithReorient()
        image, _ = reader.read_images([str(Path(cfg["data"]["image_dir"]) / f"{case_id}.nii.gz")])
        gt, _ = reader.read_seg(str(gt_file))
        if gt.ndim == 4 and gt.shape[0] == 1:
            gt = gt[0]
        _, gt, _ = crop_to_nonzero(image.astype(np.float32, copy=False), gt[None])
        gt = gt[0]
        if gt.shape != original.shape:
            raise ValueError(f"GT/ROI shape mismatch for {case_id}: {gt.shape} vs {original.shape}")
        mask0 = original > threshold
        dice = hard_dice(mask0, gt > 0)
        baseline.append({"case_id":case_id,"historical_dice":hist.get(case_id,float("nan")),"current_dice":dice,
                         "absolute_difference":abs(hist.get(case_id,float("nan"))-dice),
                         "id_matched":case_id in hist,"original_foreground_voxels":int(mask0.sum()),
                         "gt_foreground_voxels":int((gt>0).sum()),"spacing":json.dumps(metadata["spacing"]),
                         "prediction_physical_volume_mm3":float(mask0.sum()*np.prod(metadata["spacing"])),
                         "gt_physical_volume_mm3":float((gt>0).sum()*np.prod(metadata["spacing"]))})
        valid = np.ones(original.shape, dtype=bool)
        masks_all = np.concatenate([mask0[None], iv > threshold, pv > threshold], axis=0)
        union = masks_all.any(axis=0)
        raw_groups = {"Image-only":iv,"Prompt-only":pv,"Image+Prompt":None}
        for name, views in raw_groups.items():
            if views is None: continue
            for metric, u in (("I1_Pairwise_Hard",1-pairwise_hard(views>threshold)),
                              ("I2_Pairwise_Soft",1-pairwise_soft(views)),("I3_VVC",vvc(views>threshold))):
                rows.append(dict(case_id=case_id,metric=metric,perturbation=name,aggregation="case",gating="raw",
                                 uncertainty=u,confidence=confidence(metric,u),valid_voxels=int(valid.sum()),dice=dice))
        for metric,u in (("I1_Pairwise_Hard",1-(pairwise_hard(iv>threshold)+pairwise_hard(pv>threshold))/2),
                         ("I2_Pairwise_Soft",1-(pairwise_soft(iv)+pairwise_soft(pv))/2),
                         ("I3_VVC",(vvc(iv>threshold)+vvc(pv>threshold))/2)):
            rows.append(dict(case_id=case_id,metric=metric,perturbation="Image+Prompt",aggregation="case",gating="raw",
                             uncertainty=u,confidence=confidence(metric,u),valid_voxels=int(valid.sum()),dice=dice))
        groups = [("Image-only",iv),("Prompt-only",pv),("Image+Prompt",None)]
        for perturb, views in groups:
            maps = uncertainty_maps(original, np.concatenate((iv,pv),axis=0) if views is None else views, threshold)
            for metric, map_ in maps.items():
                for agg, region in (("G0",None),("G1",union)):
                    u,n = aggregate(map_,valid,region)
                    rows.append(dict(case_id=case_id,metric=metric,perturbation=perturb,aggregation=agg,gating="raw",
                                     uncertainty=u,confidence=confidence(metric,u),valid_voxels=n,dice=dice))
        # Cross-modal disagreement is defined between the two view means.
        ipd = np.abs(iv.mean(axis=0)-pv.mean(axis=0))
        for agg, region in (("G0",None),("G1",union)):
            u,n = aggregate(ipd,valid,region)
            rows.append(dict(case_id=case_id,metric="I13_IPD",perturbation="Image+Prompt",aggregation=agg,gating="raw",
                             uncertainty=u,confidence=confidence("I13_IPD",u),valid_voxels=n,dice=dice))
        # I0 is the exact original A score; I12 uses the repository's Chebyshev
        # map and converts confidence to uncertainty for consistent reporting.
        if case_id in a_scores:
            u = 1-a_scores[case_id]
            rows.append(dict(case_id=case_id,metric="I0_CUE_DIC",perturbation="Original-A",aggregation="case",gating="raw",
                             uncertainty=u,confidence=1-u,valid_voxels=int(valid.sum()),dice=dice))
        all_views = np.concatenate((iv,pv),axis=0).astype(np.float64)
        pmean=all_views.mean(axis=0); var=all_views.var(axis=0)
        cheb=var/(var+(pmean-threshold)**2+1e-8)
        cheb_u,n=aggregate(cheb,valid,None)
        rows.append(dict(case_id=case_id,metric="I12_Chebyshev",perturbation="Image+Prompt",aggregation="G0",gating="raw",
                         uncertainty=cheb_u,confidence=1-cheb_u,valid_voxels=n,dice=dice))
    # Add unified empty-prediction gates without overwriting raw observations.
    empty_ids={r["case_id"] for r in baseline if r["original_foreground_voxels"]==0}
    for row in list(rows):
        gated=dict(row); gated["gating"]="gated"
        if row["case_id"] in empty_ids:
            gated["confidence"]=-1.0; gated["uncertainty"]=float("nan")
        rows.append(gated)
    write_csv(output/"baseline_reproducibility.csv",baseline)
    write_csv(output/"per_case_scores.csv",rows)
    summarize(output,rows,baseline)


def write_csv(path, rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(output, rows, baseline):
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    from collections import defaultdict
    grouped=defaultdict(list)
    for row in rows:
        if row["gating"]=="raw":grouped[(row["metric"],row["perturbation"],row["aggregation"])].append(row)
    corr=[]; fail=[]
    for (metric,pert,agg), data in grouped.items():
        y=np.array([r["dice"] for r in data]); c=np.array([r["confidence"] for r in data]); u=np.array([r["uncertainty"] for r in data])
        finite=np.isfinite(c)&np.isfinite(y); y,c,u=y[finite],c[finite],u[finite]
        if len(y)<2: continue
        def stat(fn):
            try:
                value=fn(y,c)
                return float(value.statistic if hasattr(value,"statistic") else value[0])
            except Exception:return float("nan")
        corr.append(dict(metric=metric,perturbation=pert,aggregation=agg,n=len(y),spearman=stat(spearmanr),
                         pearson=stat(pearsonr),kendall_tau_b=stat(kendalltau)))
        failed=(y<0.1).astype(int); quality=(y>0.5).astype(int)
        for label,target,score in (("dice_lt_0.1",failed,u),("dice_gt_0.5",quality,c)):
            auc=roc_auc_score(target,score) if len(np.unique(target))==2 else float("nan")
            ap=average_precision_score(target,score) if len(np.unique(target))==2 else float("nan")
            fail.append(dict(metric=metric,perturbation=pert,aggregation=agg,target=label,auroc=auc,auprc=ap))
    write_csv(output/"correlation_summary.csv",corr);write_csv(output/"failure_detection_summary.csv",fail)
    # Baseline reproduction audit has an explicit stop condition.
    values=[r["absolute_difference"] for r in baseline if np.isfinite(r["absolute_difference"])]
    summary={"cases":len(baseline),"historical_cases":len({r["case_id"] for r in baseline if r["id_matched"]}),
             "max_abs_dice_difference":max(values) if values else None,
             "mean_abs_dice_difference":float(np.mean(values)) if values else None,
             "within_1e-4":bool(values and max(values)<=1e-4),
             "raw_empty":sum(r["original_foreground_voxels"]==0 for r in baseline),
             "raw_nonempty":sum(r["original_foreground_voxels"]>0 for r in baseline),
             "dice_zero_nonempty":sum(r["current_dice"]==0 and r["original_foreground_voxels"]>0 for r in baseline),
             "dice_lt_0_1":sum(r["current_dice"]<0.1 for r in baseline),
             "dice_0_1_to_0_5":sum(0.1<=r["current_dice"]<=0.5 for r in baseline),
             "dice_gt_0_5":sum(r["current_dice"]>0.5 for r in baseline)}
    (output/"baseline_reproducibility_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (output/"REPORT.md").write_text("# Stage 1 image-level uncertainty benchmark\n\n"
        "This report is generated from the saved per-case scores. Baseline Dice must match the historical reference within 1e-4 before metric comparisons are considered protocol-equivalent.\n\n"
        f"- Cases scored: {summary['cases']}\n- Baseline maximum absolute Dice difference: {summary['max_abs_dice_difference']}\n"
        f"- Baseline mean absolute Dice difference: {summary['mean_abs_dice_difference']}\n- Empty/nonempty original predictions: {summary['raw_empty']}/{summary['raw_nonempty']}\n"
        f"- Dice-zero nonempty predictions: {summary['dice_zero_nonempty']}\n\n"
        "Full comparison tables are in `correlation_summary.csv`, `failure_detection_summary.csv`, and `per_case_scores.csv`."
        " Historical A scores are copied from the original confidence partition; E routing metadata is retained in the manifest.\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/response_routing_liver_cancer_E.yaml")
    parser.add_argument("--output-dir",default="analysis/hierarchical_uncertainty/stage1_image_level")
    parser.add_argument("--gt-dir",default="/data/hilab/datasets/WCH_MRI145/DWI/labels")
    parser.add_argument("--reference-dir",default="/home/ubuntu/zy/pythoncode/sfda/CUE-SFDA")
    parser.add_argument("--devices",type=int,nargs="+",default=[0,1])
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--limit-cases",type=int)
    parser.add_argument("--worker-rank",type=int)
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    # Use the E runtime's default fp16 text encoder. The E standard-prompt
    # embedding cache is copied read-only into a benchmark-owned cache folder.
    cfg["model"]["text_encoder_dtype"] = "float16"
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    (output/"resolved_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    (output/"experiment_manifest.json").write_text(json.dumps({"branch":"image-level-uncertainty-benchmark","config":str(Path(args.config).resolve()),
        "seed":cfg["experiment"]["seed"],"devices":args.devices,"image_view_count":cfg["method"]["num_views"],
        "prompts":cfg["method"]["routing_prompts"],"standard_prompt":cfg["model"]["prompt"],"gt_role":"offline_evaluation_only",
        "gt_dir":str(Path(args.gt_dir).resolve()),"reference_dir":str(Path(args.reference_dir).resolve()),
        "model_checkpoint":str(Path(cfg["model"]["model_dir"]).resolve()/"checkpoint_final.pth"),
        "text_encoder_dtype":"float16","original_prompt_strategy":"standard prompt only, single prompt forward",
        "probability_threshold":float(cfg["method"]["probability_threshold"]),
        "gaussian_fusion":{"sigma_scale":0.125,"value_scaling_factor":10,"logit_fusion":True,"tile_step_size":float(cfg["train"]["tile_step_size"])},
        "uncertainty_map_epsilon":1e-8,"epkl_probability_epsilon":1e-6,
        "confidence_transforms":{"I0_CUE_DIC":"1-U","I1_Pairwise_Hard":"1-U","I2_Pairwise_Soft":"1-U","I3_VVC":"1/(1+U)",
          "I4_MPC":"1-2U","I5_Entropy":"1-U","I6_PE":"1-U","I7_EE":"1-U","I8_PV":"1-4U","I9_MI":"1-U",
          "I10_EPKL":"1/(1+U)","I11_VD":"1-2U","I12_Chebyshev":"1-U","I13_IPD":"1/(1+U)"},
        "existing_multiview_probability_cache":"none found; inference required","historical_E":str(Path("outputs/response_routing/E_joint_empty_to_low/routing_summary.json").resolve())},indent=2)+"\n")
    if args.worker_rank is not None:
        case_worker(args,cfg,output,args.worker_rank,len(args.devices));return
    if len(args.devices)>1:
        procs=[]
        for rank in range(len(args.devices)):
            command=[sys.executable,str(Path(__file__).resolve()),"--config",args.config,"--output-dir",str(output),"--gt-dir",args.gt_dir,
                     "--reference-dir",args.reference_dir,"--devices",*map(str,args.devices),"--worker-rank",str(rank)]
            if args.force:command.append("--force")
            if args.limit_cases is not None:command.extend(["--limit-cases",str(args.limit_cases)])
            procs.append(subprocess.Popen(command,cwd=ROOT))
        statuses=[p.wait() for p in procs]
        if any(statuses):raise SystemExit(f"Inference workers failed: {statuses}")
    else:
        case_worker(args,cfg,output,0,1)
    # Do not score a partial run.
    if args.limit_cases is not None:
        return
    expected=len(load_split(cfg["data"]["image_dir"],cfg["data"]["split_json"])[0])
    found=len(list((output/"probabilities").glob("*.npz")))
    if found!=expected:raise RuntimeError(f"Incomplete inference cache: expected {expected} cases, found {found}")
    score_all(args,cfg,output)


if __name__=="__main__":main()
