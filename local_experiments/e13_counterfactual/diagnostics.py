import json,concurrent.futures
import numpy as np
import torch
from scipy.stats import rankdata,spearmanr
from .core import *
from .train import infer
from .evaluate import NAMES

def one(eid):
 torch.set_num_threads(1);d=np.load(OUT/'events'/eid/'anchors.npz');G=d['gains'];F=d['features'];truth=rankdata(-G,axis=1);delta=truth[1:]-truth[:1];rows=[];predictions={}
 for model in NAMES:
  score=infer(OUT/'models'/model/'weights.npz',F);predictions[model]=score;rank=rankdata(-score,axis=1);diff=rank[1:]-rank[:1]
  corr=spearmanr(diff.ravel(),delta.ravel()).statistic if np.ptp(diff)>0 and np.ptp(delta)>0 else np.nan
  correlations=[]
  for s in range(len(G)):
   v=spearmanr(score[s],G[s]).statistic
   if np.isfinite(v):correlations.append(float(v))
  if model.startswith('A0'):assert np.array_equal(rank, np.broadcast_to(rank[:1],rank.shape))
  rows.append(dict(event_id=eid,split=split(eid),model=model,mean_anchor_rank_spearman=float(np.mean(correlations)) if correlations else None,conditional_rank_change_correlation=float(corr) if np.isfinite(corr) else None,predicted_rank_changes=bool(np.ptp(diff)>0)))
 # Counterfactual pairs actually repeated inside the TRAINED Medium pools.
 s=np.load(OUT/'events'/eid/'states.npz');pairs={}
 for i in range(len(s['names'])):
  for r,a,g in zip(s['remove'][i],s['add'][i],s['gains'][i]):pairs.setdefault((int(r),int(a)),[]).append(float(g))
 repeated=[g for g in pairs.values() if len(g)>1];coverage=sum(len(g) for g in repeated)/(len(s['names'])*512);flips=sum(min(g)<-1e-12 and max(g)>1e-12 for g in repeated)
 dest=OUT/'conditional_predictions';dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'{eid}.npz',**predictions)
 return dict(event_id=eid,split=split(eid),metrics=rows,medium_unique_pairs=len(pairs),medium_repeated_pairs=len(repeated),medium_repeated_row_fraction=coverage,medium_repeated_sign_flip_fraction=flips/len(repeated) if repeated else None)
if __name__=='__main__':
 ids=sorted(p.name for p in (OUT/'events').iterdir())
 with concurrent.futures.ProcessPoolExecutor(max_workers=12) as ex:rows=list(ex.map(one,ids))
 dump(OUT/'conditional_rank_metrics.json',rows);summary={}
 for sp in ('train','validation','evaluation'):
  group=[r for r in rows if r['split']==sp];summary[sp]=dict(repeated_row_fraction=float(np.mean([r['medium_repeated_row_fraction'] for r in group])),repeated_pair_sign_flip_fraction=float(np.mean([r['medium_repeated_sign_flip_fraction'] for r in group])),models={})
  for m in NAMES:
   sub=[a for r in group for a in r['metrics'] if a['model']==m];corr=[r['conditional_rank_change_correlation'] for r in sub if r['conditional_rank_change_correlation'] is not None];summary[sp]['models'][m]=dict(mean_anchor_rank_spearman=float(np.mean([r['mean_anchor_rank_spearman'] for r in sub if r['mean_anchor_rank_spearman'] is not None])),mean_conditional_rank_change_correlation=float(np.mean(corr)) if corr else None,events_with_predicted_rank_change=sum(r['predicted_rank_changes'] for r in sub))
 dump(OUT/'conditional_rank_summary.json',summary);print(json.dumps(summary,indent=2))
