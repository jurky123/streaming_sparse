import json,time
import numpy as np
import torch
from scipy.stats import rankdata,spearmanr
from .core import *
from .train import load,infer,metrics
NAMES=['A0_Risk','A2_Risk','A0_CF','A2_CF']
def freeze():
 path=OUT/'evaluation_lock.json'
 if path.exists():return json.loads(path.read_text())
 tests=[]
 for arch in ('A0','A2'):
  a=json.loads((OUT/'models'/f'{arch}_CF'/'development.json').read_text())['validation']['risk'];b=json.loads((OUT/'models'/f'{arch}_Risk'/'development.json').read_text())['validation']['risk'];delta=a['positive_filtered_mean_pct']-b['positive_filtered_mean_pct'];passed=a['positive_filtered_recovery']>=.3 and delta>=1.
  tests.append(dict(arch=arch,positive_filtered_recovery=a['positive_filtered_recovery'],positive_filtered_gain_improvement_pp=delta,passed=passed))
 passed=[x for x in tests if x['passed']];winner=max(passed,key=lambda x:x['positive_filtered_recovery'])['arch']+'_CF' if passed else None
 result=dict(frozen_at=time.time(),models={n:sha(OUT/'models'/n/'weights.npz') for n in NAMES},gating_tests=tests,gating_winner=winner,gating_status='not_run_ranking_gate_failed' if winner is None else 'requires_gating_training_before_test',test_not_accessed=True)
 if winner is not None:dump(OUT/'gating_trigger.json',result);raise RuntimeError('Ranking gate passed: run prescribed OOF gating before evaluation')
 dump(path,result);return result

def run():
 torch.set_num_threads(1);lock=freeze();ids,X,g,j,meta,counts=load('evaluation');allrows=[];summary=[]
 for name in NAMES:
  assert sha(OUT/'models'/name/'weights.npz')==lock['models'][name];scores=infer(OUT/'models'/name/'weights.npz',X);dest=OUT/'models'/name;np.savez_compressed(dest/'evaluation_scores.npz',scores=scores);dump(dest/'evaluation_index.json',meta)
  for panel,keep in [('initial_risk',np.array([m['state']==0 for m in meta])),('all_counterfactual_states',np.ones(len(meta),bool))]:summary.append(dict(model=name,panel=panel,**metrics(scores[keep],g[keep],j[keep])))
  chosen=scores.argmax(1)
  for i,m in enumerate(meta):
   a=int(chosen[i]);ds=np.load(OUT/'events'/m['event_id']/'states.npz');s=m['state'];r=int(ds['remove'][s,a]);add=int(ds['add'][s,a]);x=ds['masks'][s].copy();x[r]=False;x[add]=True;assert x[int(ds['own'])] and x.sum()==int(ds['K']);gain=float(g[i,a]);order=rankdata(-g[i],method='average');corr=spearmanr(scores[i],g[i]).statistic
   allrows.append(dict(model=name,**m,risk_or_state_nmse=float(j[i]),nmse_after=float(j[i]-gain),gain=gain,gain_pct=100*gain/j[i],positive_filtered_gain_pct=100*max(gain,0)/j[i],oracle_gain=float(max(g[i].max(),0)),chosen_rank=float(order[a]),rank_percentile=float((order[a]-1)/511),rank_spearman=float(corr) if np.isfinite(corr) else None,remove=r,add=add,selected=np.flatnonzero(x).tolist()))
 dump(OUT/'evaluation_summary.json',summary);dump(OUT/'evaluation_per_state.json',allrows);print(json.dumps([s for s in summary if s['panel']=='initial_risk'],indent=2))
if __name__=='__main__':run()
