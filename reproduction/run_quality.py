"""Reproduce paired dense/sparse early-window metrics with published weights."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['summary_pretrained','full_calibrated','mass_route','no_grouping','no_detail','projected_mean','no_new_training','downstream_router_pilot','single_step_flow']
def commands(case,seed,variant,chunks):
 common=['--case',case,'--seed',str(seed),'--chunks',str(chunks)]
 dense=['local_experiments.gap_review_v1_video',*common,'--method','dense']
 if variant=='full_calibrated':
  sparse=['local_experiments.gap_review_v1_video',*common,'--method','candidate','--checkpoint','artifacts/review_v1_20261102/route_expanded/model.pt']
 elif variant=='downstream_router_pilot':sparse=['local_experiments.gap_review_v1_route_downstream_video',*common,'--variant','full']
 elif variant=='single_step_flow':sparse=['local_experiments.gap_flow_single_video',*common,'--variant','full']
 else:sparse=['local_experiments.gap_review_v1_ablation_video',*common,'--variant','no_downstream' if variant=='summary_pretrained' else variant]
 return [dense,sparse]
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',choices=['horse_field','traffic_crossing'],default='horse_field');p.add_argument('--seed',type=int,default=101);p.add_argument('--variant',choices=VARIANTS,default='summary_pretrained');p.add_argument('--chunks',type=int,choices=[9,41],default=9);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 from restore_weights import restore
 if not a.dry_run:restore(ROOT,verify_only=True)
 env=os.environ.copy();env['GAP_TAIL_MODE']='1'
 for cmd in commands(a.case,a.seed,a.variant,a.chunks):
  argv=[sys.executable,'-u','-m',*cmd];print(' '.join(argv),flush=True)
  if not a.dry_run:subprocess.run(argv,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
