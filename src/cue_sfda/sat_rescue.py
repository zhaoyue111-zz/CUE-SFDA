from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from cue_sfda.evaluation import predict_volume, binary_metrics_full

def evaluate_sat_assisted(config, runtime, cases, gt_path, proposal_dir, threshold, output_dir, logger, save_predictions=True):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); po=out/'voxtell_only_predictions'; pa=out/'sat_assisted_predictions'
    if save_predictions: po.mkdir(exist_ok=True); pa.mkdir(exist_ok=True)
    reader=NibabelIOWithReorient(); normalizer=ZScoreNormalization(intensityproperties={}); rows=[]
    runtime.model.eval()
    for case in cases:
        image,props=reader.read_images([str(case.image_path)]); gt,_=reader.read_seg(str(Path(gt_path)/case.name)); gt=gt==int(config['evaluation']['label_value'])
        prob=predict_volume(runtime,image,normalizer,config); vpred=prob>=float(config['evaluation']['threshold'])
        p=Path(proposal_dir)/f'{case.name}.npy'; proposal=np.load(p).astype(bool).squeeze() if p.is_file() else np.zeros_like(prob,dtype=bool)
        if proposal.shape!=prob.shape: proposal=np.zeros_like(prob,dtype=bool)
        max_response=float(prob[proposal].max()) if proposal.any() else 0.0
        gate=bool((not vpred.any()) and proposal.any() and max_response>float(threshold))
        apred=proposal if gate else vpred
        vm=binary_metrics_full(vpred,gt); am=binary_metrics_full(apred,gt)
        if save_predictions:
            reader.write_seg(vpred.astype(np.uint8),str(po/case.name),props); reader.write_seg(apred.astype(np.uint8),str(pa/case.name),props)
        rows.append({'case_id':case.name,'no_response':not bool(vpred.any()),'sat_gate_pass':gate,'sat_volume':int(proposal.sum()),'voxtell_response_max_in_sat':max_response,'voxtell_only_dice':vm['dice'],'voxtell_only_iou':vm['iou'],'voxtell_only_precision':vm['precision'],'voxtell_only_recall':vm['recall'],'sat_assisted_dice':am['dice'],'sat_assisted_iou':am['iou'],'sat_assisted_precision':am['precision'],'sat_assisted_recall':am['recall'],'delta_dice':am['dice']-vm['dice']})
    with (out/'per_case.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def mean(k): return float(np.mean([r[k] for r in rows]))
    return {'mean_dice':mean('voxtell_only_dice'),'mean_iou':mean('voxtell_only_iou'),'mean_precision':mean('voxtell_only_precision'),'mean_recall':mean('voxtell_only_recall'),'assisted_mean_dice':mean('sat_assisted_dice'),'assisted_mean_iou':mean('sat_assisted_iou'),'assisted_mean_precision':mean('sat_assisted_precision'),'assisted_mean_recall':mean('sat_assisted_recall'),'no_response':sum(r['no_response'] for r in rows),'sat_gate_pass':sum(r['sat_gate_pass'] for r in rows)}
