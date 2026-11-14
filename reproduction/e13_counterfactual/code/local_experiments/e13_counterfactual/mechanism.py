import concurrent.futures
import numpy as np
from scipy.stats import rankdata,spearmanr
from .core import *

def one(eid):
 d=np.load(OUT/'events'/eid/'anchors.npz');g=d['gains'];grand=g.mean();pair=g.mean(0);sets=g.mean(1);inter=g-pair[None]-sets[:,None]+grand;total=float(np.mean((g-grand)**2));pv=float(np.mean((pair-grand)**2));sv=float(np.mean((sets-grand)**2));iv=float(np.mean(inter**2));assert abs(total-pv-sv-iv)<max(1e-18,total*1e-10)
 flips=((g>1e-12).any(0)&(g< -1e-12).any(0));upper=np.triu(np.ones((g.shape[1],g.shape[1]),bool),1);base=g[0,:,None]-g[0,None,:];rates=[];statepairs=[]
 for s in range(1,len(g)):
  delta=g[s,:,None]-g[s,None,:];valid=upper&(abs(base)>1e-12)&(abs(delta)>1e-12);flipped=valid&(base*delta<0);rate=float(flipped.sum()/valid.sum()) if valid.any() else None;rates.append(rate);statepairs.append(dict(state=s,non_tied_action_pairs=int(valid.sum()),ranking_flips=int(flipped.sum()),flip_rate=rate,sign_flips_from_risk=int(((g[0]*g[s]<0)&(abs(g[0])>1e-12)&(abs(g[s])>1e-12)).sum())))
 return dict(event_id=eid,split=split(eid),states=len(g),anchors=g.shape[1],total_variance=total,pair_main_fraction=pv/total if total>0 else None,set_main_fraction=sv/total if total>0 else None,interaction_fraction=iv/total if total>0 else None,state_dependent_fraction=(sv+iv)/total if total>0 else None,anchor_sign_flip_fraction=float(flips.mean()),mean_rank_flip_vs_risk=float(np.mean([r for r in rates if r is not None])),state_pairs=statepairs,max_abs_gain_change=float(np.max(np.abs(g-g[0]))))
if __name__=='__main__':
 ids=sorted(p.name for p in (OUT/'events').iterdir())
 with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:rows=list(ex.map(one,ids))
 dump(OUT/'mechanism_per_event.json',rows);summary={}
 for sp in ('train','validation','evaluation'):
  rr=[r for r in rows if r['split']==sp];summary[sp]=dict(events=len(rr),states=sum(r['states'] for r in rr),anchors=sum(r['anchors'] for r in rr),**{k+'_mean':float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ('pair_main_fraction','set_main_fraction','interaction_fraction','state_dependent_fraction','anchor_sign_flip_fraction','mean_rank_flip_vs_risk')})
 dump(OUT/'mechanism_summary.json',summary);print(json.dumps(summary,indent=2))
