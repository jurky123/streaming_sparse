"""E15: frozen-summary diagnostics, no fitted parameters or new sampling."""
import os,json,time,csv,re,concurrent.futures,tarfile,shutil
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from ..e14_value_observability.core import E13,DATA,load_online,geometry,cosine,split,sha,dump
ROOT=Path(__file__).resolve().parents[2]
E14=ROOT/'artifacts/e14_value_observability_v1'
OUT=ROOT/'artifacts/e15_summary_information_v1'
def corr(x,y,rank=False):
 x=np.asarray(x).ravel();y=np.asarray(y).ravel()
 if np.std(x)<1e-25 or np.std(y)<1e-25:return None
 return float(spearmanr(x,y).statistic if rank else np.corrcoef(x,y)[0,1])
def mean_cos(a,b):
 c,v=cosine(a,b);return float(c[v].mean()) if v.any() else None

def proxies(f):
 q=f['query'][f['probe_indices'].astype(int)].astype(float);k=f['summary_keys'].astype(float);U=f['basis'].astype(float);mu=f['summary_mean'].astype(float)
 log=np.einsum('qd,prd->qpr',q,k)/np.sqrt(128);w=np.exp(log-log.max((1,2),keepdims=True));mass=w.sum(-1);mass/=mass.sum(1,keepdims=True);within=w/w.sum(-1,keepdims=True)
 haar=np.array([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],float)
 det=np.einsum('ar,pad->prd',haar,f['summary_detail'].astype(float)@U.T)
 v=mu[None]+np.einsum('qpr,prd->qpd',within,det)
 full=mass[...,None]*(v-np.einsum('qp,qpd->qd',mass,v)[:,None])
 means=mass[...,None]*(mu[None]-np.einsum('qp,pd->qd',mass,mu)[:,None])
 return {'rank8':full@U,'full_v':full,'mean_only':means},U

def one(eid):
 dest=OUT/'events'/eid;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 st=np.load(E13/'events'/eid/'states.npz');an=np.load(E13/'events'/eid/'anchors.npz');raw=np.load(DATA/'contributions'/eid/'residuals_fp64.npz')
 f=load_online(DATA/'contributions'/eid);G=an['gains'];r=an['remove'];a=an['add'];masks=st['masks'];J=st['nmse'];names=st['names'];S,A=G.shape
 assert np.all(masks[:,r]) and not np.any(masks[:,a]) and np.all(masks[:,int(st['own'])])
 ident=dict(event_id=eid,split=split(eid));layer=int(re.search(r'_l(\d+)',eid)[1]);chunk=int(re.search(r'_c(\d+)',eid)[1]);step=int(re.search(r'_s(\d+)_h',eid)[1])
 jm=int(G.mean(0).argmax());jp=int(np.maximum(G,0).mean(0).argmax());best=np.maximum(G.max(1),0);bj=G.argmax(1);vr=best-np.maximum(G[:,jm],0);vs=best-np.maximum(G[:,jp],0)
 rows=[]
 for s in range(S):
  j=int(bj[s]);noop=bool(best[s]<=1e-12);rh=bool(f['page_is_history'][r[j]]);ah=bool(f['page_is_history'][a[j]])
  rows.append(ident|dict(state=str(names[s]),state_index=s,layer=layer,chunk=chunk,step=step,nmse=float(J[s]),conditional_gain=float(best[s]),fixed_requested=float(max(0,G[s,jm])),fixed_strong=float(max(0,G[s,jp])),V_requested=float(vr[s]),V_strong=float(vs[s]),event_weight=1/S,material_requested=bool(vr[s]>=max(1e-12,.01*J[s])),material_strong=bool(vs[s]>=max(1e-12,.01*J[s])),best_action=-1 if noop else j,initial_best_changed=bool((j!=bj[0]) and not noop),initial_action_regret=float(best[s]-max(0,G[s,bj[0]])),exchange_type='noop' if noop else ('H' if rh else 'C')+'->'+('H' if ah else 'C'),remove_age=int(f['page_age'][r[j]]),add_age=int(f['page_age'][a[j]])))
 ps,U=proxies(f);mass,h,_=geometry(f);assert np.allclose(ps['rank8'],mass[...,None]*h,atol=1e-12,rtol=1e-10)
 probe=f['probe_indices'].astype(int);e=raw['e'][probe];E=np.stack([e[:,~x].sum(1) for x in masks]);de=e[:,r]-e[:,a];results=[];actions=[];scores={};ii,jj=np.triu_indices(S,1);truth=G[jj]-G[ii];centered=truth-truth.mean(-1,keepdims=True)
 for name,hp in ps.items():
  H=np.stack([hp[:,x].sum(1) for x in masks]);v=hp[:,r]-hp[:,a]
  # Fixed proxy Ehat=-H. Swap residual Ehat'=Ehat+v. Equal denominator, no teacher scale.
  score=(2*np.einsum('sqd,qad->sa',H,v)-np.square(v).sum(-1).sum(0)[None])/len(probe)
  scores[name]=score;ds=score[jj]-score[ii];dc=ds-ds.mean(-1,keepdims=True)
  H128=H@np.linalg.pinv(U) if name=='rank8' else H;v128=v@np.linalg.pinv(U) if name=='rank8' else v
  selected=score.argmax(1);g=G[np.arange(S),selected];gate=score.max(1)>0;actual=np.where(gate,g,0);pos=np.maximum(g,0)
  rankrho=[corr(score[s],G[s],True) for s in range(S)]
  row=ident|dict(representation=name,set_full_cosine=mean_cos(-H128,E),candidate_full_cosine=mean_cos(v128,de),set_coordinate_cosine=mean_cos(-H,E@U) if name=='rank8' else mean_cos(-H,E),candidate_coordinate_cosine=mean_cos(v,de@U) if name=='rank8' else mean_cos(v,de),gain_change_pearson=corr(ds,truth),gain_change_spearman=corr(ds,truth,True),interaction_pearson=corr(dc,centered),interaction_spearman=corr(dc,centered,True),mean_rank_spearman=float(np.mean([v for v in rankrho if v is not None])),forced_gain=float(g.mean()),forced_relative_pct=float(np.mean(g/J)*100),positive_filtered_gain=float(pos.mean()),positive_filtered_relative_pct=float(np.mean(pos/J)*100),proxy_gate_gain=float(actual.mean()),proxy_gate_relative_pct=float(np.mean(actual/J)*100),gate_degradation_fraction=float(np.mean(actual< -1e-12)),oracle_gain=float(best.mean()),top1_hit=float(np.mean(selected==bj)),centering_residual=float(np.abs(hp.sum(1)).max()))
  results.append(row)
  for s in range(S):actions.append(ident|dict(representation=name,state=str(names[s]),state_index=s,action=int(selected[s]),predicted_score=float(score[s,selected[s]]),proxy_gate=bool(gate[s]),true_gain=float(g[s]),nmse=float(J[s]),oracle_gain=float(best[s])))
 # Independent direct FP64 swap objective on each representation's chosen masks.
 from ..e12_swap_prediction.core import objective
 err=0
 for s in range(S):
  for name,score in scores.items():
   j=int(score[s].argmax());x=masks[s].copy();x[r[j]]=False;x[a[j]]=True
   err=max(err,abs((objective(raw,masks[s])-objective(raw,x))-G[s,j]))
 assert err<1e-10
 np.savez_compressed(dest/'scores.npz',**scores,remove=r,add=a,state_names=names)
 result=ident|dict(A=rows,B=results,actions=actions,audit=dict(max_gain_error=err,rank8_reproduced=True,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [E13/'events'/eid/'states.npz',E13/'events'/eid/'anchors.npz',DATA/'contributions'/eid/'summary_and_route_features.npz',DATA/'contributions'/eid/'residuals_fp64.npz']}))
 dump(dest/'result.json',result);return result

def table(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def aggregate(rows,key):
 out=[]
 for sp,val in sorted({(r['split'],str(r[key])) for r in rows}):
  rr=[r for r in rows if r['split']==sp and str(r[key])==val];weights=np.array([r['event_weight'] for r in rr]);V=np.array([r['V_strong'] for r in rr]);R=np.array([r['V_requested'] for r in rr]);J=np.array([r['nmse'] for r in rr]);B=np.array([r['conditional_gain'] for r in rr]);allr=[r for r in rows if r['split']==sp];total=sum(r['V_strong']*r['event_weight'] for r in allr)
  out.append(dict(split=sp,group_by=key,group=val,states=len(rr),events=len(set(r['event_id'] for r in rr)),strong_V_mean=float(np.average(V,weights=weights)),strong_V_share=float(np.dot(weights,V)/total) if total else None,strong_relative_pp=float(np.average(V/J,weights=weights)*100),requested_relative_pp=float(np.average(R/J,weights=weights)*100),strong_over_oracle=float(np.dot(weights,V)/np.dot(weights,B)) if np.dot(weights,B)>0 else None))
 return out

def finish(results):
 A=[x for r in results for x in r['A']];B=[x for r in results for x in r['B']];acts=[x for r in results for x in r['actions']]
 table('A_states.csv',A);table('B_events.csv',B);table('B_actions.csv',acts)
 groups=[x for key in ['state','layer','chunk','step','exchange_type','remove_age','add_age'] for x in aggregate(A,key)];table('A_groups.csv',groups)
 sums=[]
 for sp,rep in sorted({(r['split'],r['representation']) for r in B}):
  rr=[r for r in B if (r['split'],r['representation'])==(sp,rep)];row=dict(split=sp,representation=rep,events=len(rr))
  for k in rr[0]:
   if k in ['split','representation','event_id']:continue
   vals=[r[k] for r in rr if r[k] is not None];row[k]=float(np.mean(vals)) if vals else None
  row['pooled_positive_recovery']=sum(r['positive_filtered_gain'] for r in rr)/sum(r['oracle_gain'] for r in rr)
  sums.append(row)
 table('B_summary.csv',sums)
 concentration=[]
 for sp in ['train','validation','evaluation']:
  rr=[r for r in A if r['split']==sp];w=np.array([r['event_weight'] for r in rr]);v=np.array([r['V_strong'] for r in rr]);vr=np.array([r['V_requested'] for r in rr]);b=np.array([r['conditional_gain'] for r in rr]);m=np.array([r['material_requested'] for r in rr]);ms=np.array([r['material_strong'] for r in rr]);concentration.append(dict(split=sp,states=len(rr),events=len(set(r['event_id'] for r in rr)),requested_over_conditional=float(w@vr/(w@b)),strong_over_conditional=float(w@v/(w@b)),material_requested_weight=float(w@m/w.sum()),material_requested_share_of_requested_V=float(w@(vr*m)/(w@vr)),material_requested_share_of_strong_V=float(w@(v*m)/(w@v)),material_strong_weight=float(w@ms/w.sum()),material_strong_share_of_strong_V=float(w@(v*ms)/(w@v))))
 dump(OUT/'summary.json',dict(concentration=concentration,B=sums,A_groups=groups,audit=dict(events=len(results),states=len(A),max_gain_error=max(r['audit']['max_gain_error'] for r in results),rank8_reproduced=True)))
 print(json.dumps(dict(concentration=concentration[-1],B=[r for r in sums if r['split']=='evaluation']),indent=2),flush=True)

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 protocol=dict(created=time.time(),scope='E15, same144 events994 sets; train/dev/evaluation unchanged; no fitted model, no QKV/video generation, no global solver',primary_A='strong fixed preference j_plus maximizing mean positive gain; retain j_mean requested comparison; all panel action choices offline oracle',A_weighting='equal event, equal states within event; stratified means and contributions retain original 1/S weights; age/type describes best action, descriptive not causal',material='V>=max(1e-12,.01*stateNMSE); report share of total absolute V as well as population fraction',B='M16 deployed probes, true gains Q64; same anchor candidate actions; rank8 existing coordinates, full128 summary including details, full128 means only; all use same summary K mass',score='fixed Ehat=-H, delta=v_r-v_a; score=mean_q(2H dot delta - norm(delta)^2); fixed unit denominator proxy, not a replacement of true varying-denominator objective; no true e/d/Dense/C/label scale enters score',evaluation='gain change Pearson/Spearman on all unordered state pairs; candidate-centered interaction correlation; forced argmax, proxy score>0 gate with no-op, and separately true-positive-filter diagnostic; no fitted calibration or sign selection',limitations='direction-only proxy, neither actual residual amplitude nor denominator errors estimated; negative result cannot prove summary information insufficiency',E14_protocol_hash=sha(E14/'protocol.json'),E13_preparation_hash=sha(E13/'preparation.json'),dataset_manifest_hash=sha(DATA/'manifest.json'))
 if not (OUT/'protocol.json').exists():dump(OUT/'protocol.json',protocol)
 ids=sorted(p.name for p in (E13/'events').iterdir())
 with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
  results=[]
  for i,r in enumerate(ex.map(one,ids),1):results.append(r);print(i,r['event_id'],flush=True)
 finish(results)
if __name__=='__main__':run()
