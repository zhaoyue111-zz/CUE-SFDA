#!/usr/bin/env python3
import argparse, json
import torch
try:
    _tvlib=torch.library.Library('torchvision','DEF'); _tvlib.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
except Exception: pass
from pathlib import Path
from cue_sfda.config import load_config
from cue_sfda.data import load_split
from cue_sfda.modeling import build_runtime
from cue_sfda.engine import CUETrainer
from cue_sfda.utils import configure_logging, seed_everything
from cue_sfda.sat_rescue import evaluate_sat_assisted
from cue_sfda.engine.checkpoint import load_checkpoint

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--gt-path',required=True);ap.add_argument('--proposal-dir',required=True);ap.add_argument('--gate-csv',required=True); args=ap.parse_args()
 c=load_config(args.config); c['experiment']['output_dir']=args.output_dir; c['method']['routing_mode']='joint_gate'; c['sat_rescue']={'proposal_dir':args.proposal_dir,'gate_csv':args.gate_csv,'gate_threshold':0.01}; out=Path(args.output_dir); log=configure_logging(out); seed_everything(int(c['experiment']['seed']))
 train,test=load_split(c['data']['image_dir'],c['data']['split_json']); runtime=build_runtime(c,out/'prompt_cache')
 trainer=CUETrainer(c,runtime,train,out,log,validation_cases=test,validation_gt_path=args.gt_path); trainer.run()
 # Evaluate both complete best checkpoints on test, preserving both inference paths.
 rows={}
 for name in ['best_voxtell_only.pt','best_sat_assisted.pt']:
  p=out/'checkpoints'/name
  if not p.is_file(): continue
  load_checkpoint(p,runtime.model); rows[name]=evaluate_sat_assisted(c,runtime,test,args.gt_path,args.proposal_dir,0.01,out/'test'/name.replace('.pt',''),log,True)
 with (out/'test_summary.json').open('w') as f: json.dump(rows,f,indent=2)
 print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
