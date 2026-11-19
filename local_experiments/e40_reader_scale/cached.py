"""No raw KV access: cached development phi collision and saturation diagnostic."""
import argparse,csv,hashlib,json,math,time,collections
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'artifacts/e40_reader_scale_v1';E30=ROOT/'artifacts/e30_beta_diagnostics_v1';E38=ROOT/'artifacts/e38_action_attribution_v1';E24=ROOT/'artifacts/e24_joint_representatives_v1'
def dump(path,x):
 path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(path,rows):
 if not rows:return
 with path.open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lse(v):
 v=np.asarray(v);m=v.max();return float(m+np.log(np.exp(v-m).sum()))
def pair_count(n):return n*(n-1)//2

def run(seed):
 torch.set_num_threads(2);dest=DEST/f'cached_s{seed}';dest.mkdir(exist_ok=True);assert not (dest/'complete.json').exists();ids=json.loads((ROOT/f'artifacts/e28_local_response_v1/s{seed}/identities.json').read_text());old=json.loads((E38/f'B_s{seed}/action_margin_attribution.json').read_text());reg=json.loads((E38/f'B_s{seed}/region_attribution.json').read_text());names={r['event']:i for i,r in enumerate(old)};records=[];eventrows=[];sources=[];start=time.time()
 for idx,o in enumerate(old):
  xp=E30/f's{seed}/beta_{idx:04d}.pt';tp=E38/f'B_s{seed}/target_{idx:04d}.pt';pp=E30/f's{seed}/panel_{idx:04d}.npz';x=torch.load(xp,map_location='cpu',weights_only=True);t=torch.load(tp,map_location='cpu',weights_only=True);p=np.load(pp);ev=torch.load(E24/o['event'],map_location='cpu',weights_only=True);assert ev['split']=='development' and x['event']==o['event']==t['event']==str(p['event']);F=x['F'][...,1:].float().numpy();assert np.array_equal(F.astype('float64'),x['F'][...,1:].numpy());P=F.shape[1];beta=x['beta_true'].numpy();bp=x['betas']['Predicted'].numpy();lm=x['logmass'].numpy();response_error=(x['predicted_response'].numpy()-t['true_response'].numpy());local_mse=np.mean(response_error**2,axis=-1);q=ev['q'].float().numpy();qh=[hashlib.sha256(z.tobytes()).hexdigest() for z in q];C=np.zeros(P,bool);C[p['initial_remove']]=True;C[p['initial_add']]=True;ep=np.zeros(P,bool);pred=np.zeros(P,bool);opt=np.zeros(P,bool)
  for j,mask in [(o['j_pred'],pred),(o['j_oracle'],opt)]:
   if j>=0:mask[[p['initial_remove'][j],p['initial_add'][j]]]=True
  ep=pred|opt;flags={'all':np.ones(P,bool),'candidate':C,'endpoint_pair':ep,'other_candidate':C&~ep,'background':~C,'predicted_endpoint':pred,'oracle_endpoint':opt,'harmful_predicted_endpoint':pred&(o['proposed_gain']< -1e-12),'positive_predicted_endpoint':pred&(o['proposed_gain']>1e-12),'positive_region_attribution_endpoint':ep&(reg[idx]['phi_endpoint']>0)}
  meta={k:t['meta'][k] for k in ['seed','event','trajectory','layer','head','chunk','step','tile']}
  for role,ix in flags.items():
   if not ix.any():continue
   f=F[:,ix];err=bp[:,ix]-beta[:,ix];l=lm[:,ix];w=np.exp(2*l-lse(2*l.ravel()));row=dict(**meta,role=role,pages=int(ix.sum()),samples=int(f.shape[0]*f.shape[1]),component_abs_max=float(abs(f).max()),exact_saturated_components=int((abs(f)==1).sum()),exact_any_samples=int((abs(f)==1).any(-1).sum()),exact_all_samples=int((abs(f)==1).all(-1).sum()),near99_any_samples=int((abs(f)>=.99).any(-1).sum()),near99_all_samples=int((abs(f)>=.99).all(-1).sum()),near999_any_samples=int((abs(f)>=.999).any(-1).sum()),mean_tanh_derivative=float(np.mean(1-f.astype('float64')**2)),beta_mae=float(abs(err).mean()),beta_rmse=float(np.mean(err**2)**.5),beta_bias=float(err.mean()),mass2_beta_mse=float(np.sum(w*err**2)),local_mse=float(local_mse[:,ix].mean()));eventrows.append(row)
  records.append(dict(meta=meta,F=F,beta=beta,bp=bp,lm=lm,flags=flags,qh=qh,qnorm=np.linalg.norm(q,axis=1),hp=P-72));sources.append(dict(event=o['event'],cache_sha=sha(xp),target_sha=sha(tp),panel_sha=sha(pp),query_source_sha=sha(E24/o['event'])))
 return analyze(seed,dest,ids,names,records,eventrows,sources,start)

def analyze(seed,dest,ids,names,records,eventrows,sources,start,split='development'):
 # History identities were previously verified with token IDs, raw KV and new C summary.
 groups=[];seen=set()
 for rec in ids:
  occ=[(names[z['event']],z['page']) for z in rec['occurrences']];assert all(p<records[i]['hp'] for i,p in occ)
  for ip in occ:assert ip not in seen;seen.add(ip)
  groups.append((rec['raw_sha']+':'+rec['summary_sha'],'history',occ))
 for i,rec in enumerate(records):
  for p in range(rec['hp']):assert (i,p) in seen
  for p in range(rec['hp'],rec['F'].shape[1]):groups.append((rec['meta']['event']+':'+str(p),'current',[(i,p)]))
 collision=[];pages=[];near=[];scope_pairs=collections.Counter();total_collision_samples=0;exact_nonzero_groups=0;exact_bit_groups=0
 for gi,(identity,kind,occ) in enumerate(groups):
  f=np.concatenate([records[i]['F'][:,p] for i,p in occ]);f[f==0]=0 # +0/-0 are numerically identical for reader
  y=np.concatenate([records[i]['beta'][:,p] for i,p in occ]);pred=np.concatenate([records[i]['bp'][:,p] for i,p in occ]);lm=np.concatenate([records[i]['lm'][:,p] for i,p in occ]);ei=np.repeat([i for i,p in occ],64);qs=np.tile(np.arange(64),len(occ));qhash=np.concatenate([np.array(records[i]['qh']) for i,p in occ]);qn=np.concatenate([records[i]['qnorm'] for i,p in occ]);flags={k:np.repeat([records[i]['flags'][k][p] for i,p in occ],64) for k in records[occ[0][0]]['flags']};un,inv,count=np.unique(f,axis=0,return_inverse=True,return_counts=True);logweights=2*lm;shift=logweights.max();w=np.exp(logweights-shift);sse=0.;covered=0
  for k in np.flatnonzero(count>1):
   ix=np.flatnonzero(inv==k);ww=w[ix];den=ww.sum();mean=np.sum(ww*y[ix])/den if den else float(np.mean(y[ix]));csse=float(np.sum(ww*(y[ix]-mean)**2));sse+=csse;covered+=len(ix);uniqueq=len(set(qhash[ix]));span=float(np.ptp(y[ix]));exact_bit_groups+=1;exact_nonzero_groups+=span>1e-6
   events=np.unique(ei[ix]);chunks={records[i]['meta']['chunk'] for i in events};steps={records[i]['meta']['step'] for i in events};tile_contexts={(records[i]['meta']['chunk'],records[i]['meta']['step'],records[i]['meta']['tile']) for i in events};eventcounts=collections.Counter(ei[ix]);cs=collections.Counter((records[i]['meta']['chunk'],records[i]['meta']['step']) for i in ei[ix]);ch=collections.Counter(records[i]['meta']['chunk'] for i in ei[ix]);n=len(ix);within=sum(pair_count(v) for v in eventcounts.values());tile=sum(pair_count(v) for v in cs.values())-within;step=sum(pair_count(v) for v in ch.values())-sum(pair_count(v) for v in cs.values());chunk=pair_count(n)-sum(pair_count(v) for v in ch.values());scope_pairs.update(within_event=within,same_chunk_step_different_tile=tile,same_chunk_different_step=step,different_chunk=chunk)
   collision.append(dict(seed=seed,identity=identity,kind=kind,trajectory=records[occ[0][0]]['meta']['trajectory'],layer=records[occ[0][0]]['meta']['layer'],head=records[occ[0][0]]['meta']['head'],n=n,unique_queries=uniqueq,contexts=len(events),chunks=len(chunks),steps=len(steps),phi=un[k].tolist(),beta_range=span,beta_std=float(np.std(y[ix])),predicted_beta_range=float(np.ptp(pred[ix])),qnorm_range=float(np.ptp(qn[ix])),mass2_collision_mse=csse/den if den else None,log_mass2_weight=math.log(den)+shift if den else None,log_collision_sse=math.log(csse)+shift if csse else None,within_event_pairs=within,tile_pairs=tile,step_pairs=step,chunk_pairs=chunk,**{k+'_samples':int(v[ix].sum()) for k,v in flags.items()}))
  total_collision_samples+=covered;pages.append(dict(seed=seed,identity=identity,kind=kind,trajectory=records[occ[0][0]]['meta']['trajectory'],layer=records[occ[0][0]]['meta']['layer'],head=records[occ[0][0]]['meta']['head'],contexts=len(occ),samples=len(f),unique_phi=len(un),collision_samples=covered,normalized_collision_sse=sse/w.sum(),normalized_prediction_sse=float(np.sum(w*(pred-y)**2)/w.sum()),log_total_weight=math.log(w.sum())+shift,collision_sse_log=math.log(sse)+shift if sse else None,prediction_sse_log=math.log(np.sum(w*(pred-y)**2))+shift if np.sum(w*(pred-y)**2)>0 else None))
  # Fixed descriptive nearest-pair check, 2nd view only: no collision lower-bound claim.
  if kind=='history' and len(occ)>1:
   f64=f.astype('float64');dist=np.maximum(0,np.sum(f64*f64,1)[:,None]+np.sum(f64*f64,1)[None]-2*f64@f64.T);chunk=np.array([records[i]['meta']['chunk'] for i in ei]);step=np.array([records[i]['meta']['step'] for i in ei]);other=ei[:,None]!=ei[None];samechunk=chunk[:,None]==chunk[None];samestep=step[:,None]==step[None]
   scopes={'other_event':other,'same_chunk_step_other_tile':other&samechunk&samestep,'same_chunk_other_step':samechunk&~samestep,'other_chunk':~samechunk,'strict_past_chunk':chunk[None]<chunk[:,None]}
   for scope,allowed in scopes.items():
    dd=np.where(allowed,dist,np.inf);nn=np.argmin(dd,axis=1);z=np.flatnonzero(np.isfinite(dd[np.arange(len(f)),nn]))
    if not len(z):continue
    distance=np.linalg.norm(f64[z]-f64[nn[z]],axis=1);near.append(dict(seed=seed,identity=identity,layer=records[occ[0][0]]['meta']['layer'],scope=scope,contexts=len(occ),n=len(z),mean_phi_distance=float(distance.mean()),median_phi_distance=float(np.median(distance)),mean_beta_difference=float(abs(y[z]-y[nn[z]]).mean()),mean_qnorm_difference=float(abs(qn[z]-qn[nn[z]]).mean())))
  if gi%10000==0:print('COLLISION',seed,gi,len(groups),round(time.time()-start),flush=True)
 table(dest/'event_roles.csv',eventrows);table(dest/'collision_groups.csv',collision);table(dest/'page_collision_bounds.csv',pages);table(dest/'near_phi_descriptive.csv',near);dump(dest/'sources.json',sources);dump(dest/'complete.json',dict(seed=seed,split=split,events=len(records),physical_history_groups=len(ids),history_occurrences=len(seen),page_groups=len(groups),collision_groups=exact_bit_groups,collision_groups_target_range_gt_1e_minus6=exact_nonzero_groups,collision_samples=total_collision_samples,scope_pairs=dict(scope_pairs),raw_KV_read=False,preactivation_available=False,page_scales_available=False,training=False,test=False,seconds=time.time()-start));print('DONE CACHED',seed,exact_bit_groups,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
