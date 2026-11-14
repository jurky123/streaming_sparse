"""Append-only T3/T4 offline experiment. No teacher fields enter online_features."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,hashlib,argparse,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .common import dump,sha
from .problem import ResidualProblem
ROOT=Path(os.environ.get('SET_ORACLE_ROOT','/home/shuchenweng/jzk/streaming_sparse/artifacts/set_oracle_v1'))
OUT=ROOT/'analyses'/'t3_t4_v1'
POOLS={'small':(8,16),'medium':(16,32),'large':(32,64)}
def split(eid):
 return 'train' if eid.startswith('city_walk_s101_') else 'validation' if eid.startswith('city_walk_s303_') else 'evaluation'
def online_load(path):
 # Explicit allowlist; NPZ teacher arrays are never read here.
 keys=['query','summary_keys','summary_mean','summary_detail','basis','risk_features_q64','risk_scores','mass_scores','probe_indices','page_age','page_is_history']
 with np.load(path) as f:return {k:f[k] for k in keys}
def online_features(f,W):
 q=f['query'].astype(float);k=f['summary_keys'].astype(float);v=f['summary_mean'].astype(float)
 logits=np.einsum('qd,prd->qpr',q,k)/np.sqrt(128);mx=logits.max((1,2),keepdims=True);weight=np.exp(logits-mx);mass=weight.sum(-1);mass/=mass.sum(1,keepdims=True)
 within=weight/weight.sum(-1,keepdims=True)
 # Three stored Haar contrasts reconstruct all four representatives.
 det=f['summary_detail'].astype(float)@f['basis'].astype(float).T
 det=np.einsum('ar,pad->prd',np.array([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]),det)
 pagev=v[None]+np.einsum('qpr,prd->qpd',within,det)
 allout=np.einsum('qp,qpd->qd',mass,pagev)
 Q,P=mass.shape
 qp=np.broadcast_to((q@W)[:,None],(Q,P,W.shape[1]));kp=np.broadcast_to((k.mean(1)@W)[None],qp.shape)
 vp=np.broadcast_to((v@W)[None],qp.shape);pv=pagev@W;oo=np.broadcast_to((allout@W)[:,None],qp.shape)
 stat=np.log1p(np.maximum(f['risk_features_q64'].astype(float),0))
 age=np.broadcast_to(f['page_age'][None,:,None]/5,(Q,P,1));hist=np.broadcast_to(f['page_is_history'][None,:,None],(Q,P,1))
 feat=np.concatenate((qp,kp,vp,pv-oo,qp*kp,stat,np.log(np.maximum(mass,1e-300))[...,None],age,hist,np.ones((Q,P,1))),axis=-1)
 return feat,mass

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 spec=dict(scope=144,split={'train':'city_walk seed101 (36)','validation':'city_walk seed303 (36)','evaluation':'horse_field seeds101/303 (72), previously seen development scene'},seed=17,pools=POOLS,ranks=[4,8,16,32],query_counts=[16,64],steps=[1,2,4],T4='ridge signed residual + log exact/summary mass predictor, fixed rank16; lambdas1e-4/1e-2/1 chosen on validation only; no test tuning',T4_gate='Report T3 signal first; explicit current request also authorizes a diagnostic minimum predictor if oracle signal is weak; do not claim deployability',frozen_manifest_sha256=sha(ROOT/'manifest.json'),online_allowlist=online_load.__code__.co_consts.__repr__(),created=time.time())
 if (OUT/'protocol.json').exists():return
 dump(OUT/'protocol.json',spec)
 cov=np.zeros((128,128));n=0
 for p in sorted((ROOT/'contributions').iterdir()):
  if split(p.name)!='train':continue
  e=np.load(p/'residuals_fp64.npz')['e'].reshape(-1,128);cov+=e.T@e;n+=len(e)
 val,vec=np.linalg.eigh(cov);pc=vec[:,::-1];rng=np.random.default_rng(17);rw=np.linalg.qr(rng.normal(size=(128,128)))[0];fw=np.linalg.qr(rng.normal(size=(128,16)))[0]
 np.savez(OUT/'projections.npz',pca=pc,random=rw,feature=fw,eigenvalues=val[::-1],n=n)

def axes(x,own,risk,pool):
 r=np.flatnonzero(x);r=r[r!=own];a=np.flatnonzero(~x)
 if pool!='full':
  nr,na=POOLS[pool];r=r[np.argsort(risk[r],kind='stable')[:nr]];a=a[np.argsort(-risk[a],kind='stable')[:na]]
 return r,a
class Objective:
 def __init__(self,e,d,zh,C,precompute=True):
  self.e=e;self.d=d;self.zh=zh;self.C=C;self.diag=np.square(e).sum(-1);self.gram=e@e.transpose(0,2,1) if precompute else None
 def value(self,x):
  E=self.e[:,~x].sum(1);z=self.zh.sum(1)+self.d[:,x].sum(1)
  return float(np.sum(np.square(E).sum(1)/z**2)/self.C)
 def swaps(self,x,r,a):
  E=self.e[:,~x].sum(1);z=self.zh.sum(1)+self.d[:,x].sum(1);h=np.einsum('qd,qpd->qp',E,self.e)
  n=np.square(E).sum(-1)[:,None,None]+2*(h[:,r,None]-h[:,None,a])+self.diag[:,r,None]+self.diag[:,None,a]-2*(self.gram[:,r[:,None],a[None,:]] if self.gram is not None else self.e[:,r]@self.e[:,a].transpose(0,2,1))
  zz=z[:,None,None]-self.d[:,r,None]+self.d[:,None,a]
  assert np.all(zz>0)
  return np.sum(n/zz**2,axis=0)/self.C

def rollout(p,obj,risk,pool='full',compare=True):
 x=p.data['risk'].copy();trace=[dict(step=0,nmse=p.value(x),selected=np.flatnonzero(x).tolist())];diag=[];stopped=False
 for step in range(1,5):
  if stopped:trace.append(dict(trace[-1],step=step,held=True));continue
  r,a=axes(x,p.own,risk,pool);score=obj.swaps(x,r,a);ij=np.unravel_index(np.argmin(score),score.shape);predgain=obj.value(x)-score[ij];truebefore=p.value(x)
  if compare:
   rr,aa,true=p.swaps(x);truegain=truebefore-true;best=float(truegain.max());ri=np.searchsorted(rr,r);ai=np.searchsorted(aa,a);actual=true[np.ix_(ri,ai)];chosen=float(actual[ij]);bestpool=float(truebefore-actual.min());idx=np.unravel_index(np.argmin(true),true.shape)
   corr=spearmanr(score.ravel(),actual.ravel()).statistic if len(score.ravel())>1 else None
   diag.append(dict(step=step,best_full_gain=best,pool_best_gain=bestpool,pool_gain_ratio=bestpool/best if best>1e-12 else None,best_in_pool=bool(rr[idx[0]] in r and aa[idx[1]] in a),best_action_hit=bool(r[ij[0]]==rr[idx[0]] and a[ij[1]]==aa[idx[1]]),spearman=float(corr) if corr is not None and np.isfinite(corr) else None,predicted_gain=float(predgain),chosen_true_gain=truebefore-chosen,false_positive=bool(predgain>1e-12 and chosen>truebefore+1e-12),actions=int(len(r)*len(a))))
  if predgain<=max(1e-12,abs(obj.value(x))*1e-9):stopped=True;trace.append(dict(trace[-1],step=step,held=True));continue
  x[r[ij[0]]]=False;x[a[ij[1]]]=True
  assert x[p.own] and x.sum()==p.K
  trace.append(dict(step=step,nmse=p.value(x),remove=int(r[ij[0]]),add=int(a[ij[1]]),selected=np.flatnonzero(x).tolist(),predicted_gain=float(predgain)))
 return dict(trace=trace,diagnostics=diag)

def t3(eid):
 dest=OUT/'t3'/f'{eid}.json'
 if dest.exists():return eid
 ts=time.monotonic();p=ResidualProblem(ROOT/'contributions'/eid);f=online_load(p.root/'summary_and_route_features.npz');proj=np.load(OUT/'projections.npz');results={}
 # Full Q64 oracle is recomputed for exactly four steps and checked against T2.
 for nq in (64,16):
  qi=np.arange(64) if nq==64 else f['probe_indices'].astype(int)
  full=Objective(p.e[qi],p.d[qi],p.data['Zh'][qi],p.C)
  for pool in ('full','small','medium','large'):results[f'full_q{nq}_{pool}']=rollout(p,full,f['risk_scores'],pool)
  for kind in ('random','pca'):
   for rank in (4,8,16,32):
    W=proj[kind][:,:rank];W=W*np.sqrt(128/rank) if kind=='random' else W
    obj=Objective(p.e[qi]@W,p.d[qi],p.data['Zh'][qi],p.C)
    results[f'{kind}_r{rank}_q{nq}']=rollout(p,obj,f['risk_scores'])
 old=json.loads((ROOT/'searches/v1'/eid/'result.json').read_text())['trajectories']['risk']
 for i,row in enumerate(results['full_q64_full']['trace']):assert abs(row['nmse']-old[min(i,len(old)-1)]['nmse'])<1e-10
 dump(dest,dict(event_id=eid,split=split(eid),results=results,wall_seconds=time.monotonic()-ts));return eid

def train():
 # Fit only city s101; normalize teacher targets by ALL-SUMMARY mass, obtainable online.
 proj=np.load(OUT/'projections.npz');W=proj['feature'];V=proj['pca'][:,:16];XX=XY=None;ns=0
 for path in sorted((ROOT/'contributions').iterdir()):
  if split(path.name)!='train':continue
  f=online_load(path/'summary_and_route_features.npz');feat,mass=online_features(f,W);r=np.load(path/'residuals_fp64.npz');scale=r['summary_mass'].sum(1)
  target=np.concatenate(((r['e']@V)/scale[:,None,None],np.log(r['exact_mass']/r['summary_mass'])[...,None]),axis=-1)
  # All 64 Queries and all pages. Accumulate sufficient statistics in FP64.
  X=feat.reshape(-1,feat.shape[-1]);Y=target.reshape(-1,17)
  if XX is None:XX=np.zeros((X.shape[1],X.shape[1]));XY=np.zeros((X.shape[1],17))
  XX+=X.T@X;XY+=X.T@Y;ns+=len(X)
 norm=np.sqrt(np.diag(XX)/ns);norm=np.maximum(norm,1e-8);A=XX/norm[:,None]/norm[None,:]/ns;B=XY/norm[:,None]/ns
 for lam in (1e-4,1e-2,1.):
  reg=np.eye(len(norm))*lam;reg[-1,-1]=0;coef=np.linalg.solve(A+reg,B)/norm[:,None]
  np.savez(OUT/f'ridge_{lam}.npz',coef=coef,feature=W,pca=V,lambda_=lam,training_samples=ns)
 dump(OUT/'training.json',dict(train_instances=36,samples=ns,features=len(norm),output_rank=16,outputs=17,lambdas=[1e-4,1e-2,1.],teacher_labels_only=True))

def t4(eid):
 dest=OUT/'t4'/f'{eid}.json'
 if dest.exists():return eid
 p=ResidualProblem(ROOT/'contributions'/eid);f=online_load(p.root/'summary_and_route_features.npz');results={}
 for lam in (1e-4,1e-2,1.):
  model=np.load(OUT/f'ridge_{lam}.npz')
  for nq in (16,64):
   qi=f['probe_indices'].astype(int) if nq==16 else np.arange(64)
   fi=dict(f);fi['query']=f['query'][qi];fi['risk_features_q64']=f['risk_features_q64'][qi]
   start=time.perf_counter();feat,mass=online_features(fi,model['feature']);pred=feat@model['coef'];featsec=time.perf_counter()-start
   clipped=int(np.sum(np.abs(pred[...,-1])>20));ze=mass*np.exp(np.clip(pred[...,-1],-20,20));d=ze-mass;e=pred[...,:16]
   for pool in ('small','medium','large'):
    start=time.perf_counter();obj=Objective(e,d,mass,1.,precompute=False);build=time.perf_counter()-start
    result=rollout(p,obj,f['risk_scores'],pool)
    times=[]
    for rep in range(4):
     st=time.perf_counter();x=p.data['risk'].copy()
     for step in range(4):
      rr,aa=axes(x,p.own,f['risk_scores'],pool);scores=obj.swaps(x,rr,aa);ij=np.unravel_index(np.argmin(scores),scores.shape)
      if scores[ij]>=obj.value(x)-max(1e-12,abs(obj.value(x))*1e-9):break
      x[rr[ij[0]]]=False;x[aa[ij[1]]]=True
     if rep:times.append(time.perf_counter()-st)
    result.update(feature_predict_seconds=featsec,objective_build_seconds=build,decision_seconds_median=float(np.median(times)),clipped_logratio_count=clipped)
    results[f'ridge{lam}_q{nq}_{pool}']=result
 dump(dest,dict(event_id=eid,split=split(eid),results=results));return eid

def paired(eid):
 dest=OUT/'t3_paired'/f'{eid}.json'
 if dest.exists():return eid
 p=ResidualProblem(ROOT/'contributions'/eid);f=online_load(p.root/'summary_and_route_features.npz');W=np.load(OUT/'projections.npz')['pca'][:,:16];results={}
 for nq in (16,64):
  qi=f['probe_indices'].astype(int) if nq==16 else np.arange(64)
  obj=Objective(p.e[qi]@W,p.d[qi],p.data['Zh'][qi],p.C)
  for pool in POOLS:results[f'pca_r16_q{nq}_{pool}']=rollout(p,obj,f['risk_scores'],pool)
 dump(dest,dict(event_id=eid,split=split(eid),results=results));return eid

def queue(stage,workers):
 ids=sorted(p.name for p in (ROOT/'contributions').iterdir());fn={'t3':t3,'t4':t4,'paired':paired}[stage]
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
  jobs={ex.submit(fn,e):e for e in ids}
  for i,f in enumerate(concurrent.futures.as_completed(jobs),1):
   eid=f.result();print(stage,i,eid,flush=True);dump(OUT/f'{stage}_status.json',dict(completed=i,total=len(ids),status='complete' if i==len(ids) else 'running'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','t3','train','t4','paired']);a.add_argument('--workers',type=int,default=16);arg=a.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if arg.stage=='prepare':prepare()
 elif arg.stage=='train':train()
 else:queue(arg.stage,arg.workers)
