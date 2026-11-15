"""Replay-only Top-L recall and B-pool ceilings. No raw QKV or model calls."""
import json,csv,hashlib,math,re,time,tarfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'artifacts/e30_beta_diagnostics_v1';DEST=ROOT/'artifacts/e35_candidate_recall_v1';LS=[1,2,4,8]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(name,rows):
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def independent(x):
 q=np.linspace(0,63,16).round().astype(int)
 def arr(t):return t[q].double().numpy()
 lm=arr(x['logmass']);beta=arr(x['betas']['Predicted']);rp=arr(x['predicted_response']);center=arr(x['center']);le=lm+beta;mx=le.max(1,keepdims=True);ee=np.exp(le-mx);logD=(mx+np.log(ee.sum(1,keepdims=True)));target=((ee/ee.sum(1,keepdims=True))[...,None]*(center+rp)).sum(1);extra=max(0,float((lm-logD).max()));net=np.exp(le-logD-extra)[...,None]*(center+rp-target[:,None])-np.exp(lm-logD-extra)[...,None]*(center-target[:,None]);return (net*net).sum(-1).mean(0)
def order(score):return np.argsort(-score,kind='stable')
def metrics(g,idx,oracle):
 values=g[idx];best=max(0.,float(values.max()));return dict(best_gain=best,regret=oracle-best,has_positive=bool(values.max()>1e-12),best_hit=bool(best>=oracle-1e-12),positive_actions=int((values>1e-12).sum()),gain_min=float(values.min()),gain_p10=float(np.quantile(values,.1)),gain_median=float(np.median(values)),gain_max=float(values.max()),gain_mean=float(values.mean()),gain_std=float(values.std()))
def agg(rows):
 o=sum(r['oracle_gain'] for r in rows);g=sum(r['best_gain'] for r in rows);j=sum(r['initial'] for r in rows);valid=[r for r in rows if r['oracle_gain']>1e-12];rat=[r['best_gain']/r['oracle_gain'] for r in valid];nr=[100*r['best_gain']/r['initial'] for r in rows if r['initial']>1e-12];return dict(events=len(rows),positive_oracle_events=len(valid),oracle_gain_sum=o,best_gain_sum=g,recovery=g/o if o else None,pooled_nmse_improvement_pct=100*g/j,mean_nmse_improvement_pct=float(np.mean(nr)),positive_coverage=sum(r['has_positive'] for r in valid)/len(valid) if valid else None,positive_fraction_all=sum(r['has_positive'] for r in rows)/len(rows),best_hit_fraction=sum(r['best_hit'] for r in valid)/len(valid) if valid else None,mean_event_recovery=float(np.mean(rat)) if rat else None,median_event_recovery=float(np.median(rat)) if rat else None,mean_regret=float(np.mean([r['regret'] for r in rows])),p90_regret=float(np.quantile([r['regret'] for r in rows],.9)),max_regret=max(r['regret'] for r in rows),unique_pages_mean=float(np.mean([r['unique_pages'] for r in rows])),unique_pages_p90=float(np.quantile([r['unique_pages'] for r in rows],.9)),unique_pages_max=max(r['unique_pages'] for r in rows),actions_mean=float(np.mean([r['actions'] for r in rows])))
def run():
 assert not (DEST/'complete.json').exists();start=time.time();rows=[];actions=[];pools=[];audit=[];sources=[];baseline=[]
 for seed in [17,29]:
  old=json.loads((SRC/f's{seed}/decisions.json').read_text());old={(r['event'],r['method'],r['policy']):r for r in old};dest=DEST/f's{seed}';dest.mkdir(parents=True,exist_ok=True)
  for i in range(288):
   path=SRC/f's{seed}/panel_{i:04d}.npz';bp=SRC/f's{seed}/beta_{i:04d}.pt';p=np.load(path);x=torch.load(bp,map_location='cpu',weights_only=True);event=str(p['event']);assert event==x['event'];g=p['true_initial_gain'];score=p['initial_Predicted'];r=p['initial_remove'];a=p['initial_add'];mask=p['states'][0];assert len(g)==len(score)==512;assert np.isfinite(score).all() and np.isfinite(g).all();energy=independent(x);ind=energy[a]-energy[r];orders={'Set':order(score),'Independent':order(ind)};o=max(0.,float(g.max()));oldset=old[event,'Predicted','Set'];meta={k:oldset[k] for k in ['seed','event','trajectory','layer','head','chunk','step','tile']};meta.update(initial=oldset['initial'],oracle_gain=o);own=len(mask)-72+meta['tile'];assert mask[own] and not np.any(r==own)
   for policy,oo in orders.items():
    prior=old[event,'Predicted',policy];j=int(oo[0]);same=(int(r[j]),int(a[j]))==(prior['remove'],prior['add']);delta=abs(max(0.,float(g[j]))-prior['hindsight_gain']);assert delta<1e-10 and (same or abs(float(g[j])-prior['proposal_gain'])<1e-12);audit.append(dict(seed=seed,event=event,policy=policy,top1_same_action=same,hindsight_gain_error=delta));baseline.append(dict(**meta,policy=policy,executed_gain=prior['executed_gain'],hindsight_gain=prior['hindsight_gain']))
    for L in LS:
     idx=oo[:L];pages=np.unique(np.r_[r[idx],a[idx]]);row=dict(**meta,policy=policy,L=L,actions=L,unique_pages=len(pages),**metrics(g,idx,o));rows.append(row)
    for rank,jj in enumerate(oo[:8],1):actions.append(dict(**meta,policy=policy,rank=rank,action_index=int(jj),remove=int(r[jj]),add=int(a[jj]),predicted_score=float(score[jj] if policy=='Set' else ind[jj]),true_gain=float(g[jj])))
   proposals=np.unique(np.r_[orders['Set'][:2],orders['Independent'][:2]]);remove=np.unique(r[proposals]);add=np.unique(a[proposals]);pages=np.unique(np.r_[remove,add]);assert len(pages)<=8 and len(remove)<=4 and len(add)<=4;idx=np.flatnonzero(np.isin(r,remove)&np.isin(a,add));assert len(idx)==len(remove)*len(add)<=16
   for name,ii in [('Four-proposal-union',proposals),('Touched-page-cartesian',idx)]:
    pools.append(dict(**meta,pool=name,actions=len(ii),unique_pages=len(pages),remove_pages=len(remove),add_pages=len(add),extra_Key_bytes=len(pages)*64*128*2,extra_KV_fraction=len(pages)/(2*int(mask.sum())),hypothetical_QK_GFLOPs_layer=2*864*16*len(pages)*64*128/1e9,**metrics(g,ii,o)))
   for idxs in [orders['Set'][:8],orders['Independent'][:8],idx]:
    assert np.all(mask[r[idxs]]) and np.all(~mask[a[idxs]]) and not np.any(r[idxs]==own)
   np.savez_compressed(dest/f'event_{i:04d}.npz',event=event,mask=mask,remove=r,add=a,true_gain=g,set_score=score,independent_score=ind,set_order=orders['Set'],independent_order=orders['Independent'],proposal_union=proposals,touched_page_pool=idx)
   sources.append(dict(seed=seed,event=event,panel_sha=sha(path),beta_sha=sha(bp)))
  print('SEED DONE',seed,flush=True)
 summary=[];poolsummary=[];general=[];gates=[]
 for seed in [17,29]:
  for policy in ['Set','Independent']:
   for L in LS:
    rr=[r for r in rows if r['seed']==seed and r['policy']==policy and r['L']==L];summary.append(dict(seed=seed,policy=policy,L=L,**agg(rr)))
    if policy=='Set':
     for dim in ['trajectory','layer','chunk','step','head']:
      for v in sorted({r[dim] for r in rr}):general.append(dict(seed=seed,policy=policy,L=L,dimension=dim,value=v,**agg([r for r in rr if r[dim]==v])))
  for name in ['Four-proposal-union','Touched-page-cartesian']:
   rr=[r for r in pools if r['seed']==seed and r['pool']==name];poolsummary.append(dict(seed=seed,pool=name,**agg(rr),mean_extra_KV_fraction=float(np.mean([r['extra_KV_fraction'] for r in rr])),max_extra_KV_fraction=max(r['extra_KV_fraction'] for r in rr),mean_hypothetical_QK_GFLOPs_layer=float(np.mean([r['hypothetical_QK_GFLOPs_layer'] for r in rr]))))
  x=next(r for r in summary if r['seed']==seed and r['policy']=='Set' and r['L']==8);gates.append(dict(seed=seed,recovery_ge70=x['recovery']>=.7,positive_coverage_ge80=x['positive_coverage']>=.8,passed=x['recovery']>=.7 and x['positive_coverage']>=.8))
 for name,data in [('topL_per_event.csv',rows),('topL_summary.csv',summary),('top8_action_distribution.csv',actions),('restricted_pool_per_event.csv',pools),('restricted_pool_summary.csv',poolsummary),('generalization.csv',general),('baseline_replay.csv',audit),('baseline_per_event.csv',baseline)]:table(name,data)
 costs=[]
 for n in [2,4,8]:
  for K in [21,41]:costs.append(dict(pages=n,exact_pages=K,Key_bytes_headtile=n*64*128*2,extra_KV_fraction=n/(2*K),extra_QK_GFLOPs_layer=2*864*16*n*64*128/1e9))
 table('hypothetical_cost.csv',costs);dump(DEST/'summary.json',summary);dump(DEST/'pool_summary.json',poolsummary);dump(DEST/'gate.json',dict(results=gates,passed=all(g['passed'] for g in gates),B_executed=False,raw_Key_verifications=0));dump(DEST/'sources.json',sources);dump(DEST/'complete.json',dict(events_per_seed=288,seeds=[17,29],seconds=time.time()-start,protocol_sha=sha(DEST/'protocol.json'),network_training=False,test_accessed=False,raw_QKV_loaded=False,attention_executed=False,B_executed=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
