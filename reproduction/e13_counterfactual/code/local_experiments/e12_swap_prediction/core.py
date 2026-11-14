import os,json,hashlib,time
from pathlib import Path
import numpy as np
from ..set_oracle_v1.common import dump,sha
DATA=Path(os.environ.get('SET_ORACLE_ROOT',Path(__file__).resolve().parents[2]/'artifacts/set_oracle_v1'))
OUT=Path(os.environ.get('E12_ROOT',Path(__file__).resolve().parents[2]/'artifacts/e12_swap_prediction_v1'))
ALLOW=('query','summary_keys','summary_mean','summary_detail','basis','risk_features_probe16','risk_scores','probe_indices','page_age','page_is_history','theta')
PAGE_NAMES=['log_relative_risk','log_relative_mass','log_relative_X0','log_relative_X1','log_relative_X2','log_relative_X3','risk_rank_fraction','age_over5','history','relative_value_distance']
FEATURE_NAMES=['remove_'+x for x in PAGE_NAMES]+['add_'+x for x in PAGE_NAMES]+['set_mass','set_mass_weighted_risk','set_risk_variance','set_history_fraction']+['H_norm','H_dot_remove','H_dot_add','H_cos_remove','H_cos_add','remove_add_cos','H_norm_change','H_cos_direction_change']

def split(eid):return 'train' if eid.startswith('city_walk_s101_') else 'validation' if eid.startswith('city_walk_s303_') else 'evaluation'
def load_online(path):
 with np.load(path/'summary_and_route_features.npz') as f:return {k:f[k] for k in ALLOW}
def pages(f):
 qi=f['probe_indices'].astype(int);q=f['query'][qi].astype(float);keys=f['summary_keys'].astype(float);v=f['summary_mean'].astype(float);basis=f['basis'].astype(float)
 logits=np.einsum('qd,prd->qpr',q,keys)/np.sqrt(128);weights=np.exp(logits-logits.max((1,2),keepdims=True));mass=weights.sum(-1);mass/=mass.sum(1,keepdims=True);within=weights/weights.sum(-1,keepdims=True)
 # Haar contrasts and rank8 basis reconstruct the same deployed summary Values.
 haar=np.array([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],float);details=np.einsum('ar,pad->prd',haar,f['summary_detail'].astype(float)@basis.T)
 pv=v[None]+np.einsum('qpr,prd->qpd',within,details);allout=np.einsum('qp,qpd->qd',mass,pv);diff=pv-allout[:,None]
 risk=f['risk_scores'].astype(float);rr=risk/max(risk.mean(),1e-30);m=mass.mean(0);features=f['risk_features_probe16'].astype(float).mean(0);features=features/np.maximum(features.mean(0),1e-30)
 order=np.argsort(-risk,kind='stable');rank=np.empty(len(risk),float);rank[order]=np.arange(len(risk))/max(1,len(risk)-1)
 distance=np.sqrt(np.square(diff).sum(-1).mean(0));distance/=max(float(distance.mean()),1e-30)
 page=np.column_stack((np.log1p(rr),np.log1p(m*len(risk)),np.log1p(np.maximum(features,0)),rank,f['page_age']/5,f['page_is_history'],distance))
 H=np.einsum('qp,qpd->pd',mass,diff@basis)/len(q)
 # Six additive sufficient statistics support O(1) membership updates per action.
 additive=np.column_stack((m,m*rr,rr,rr**2,f['page_is_history'],np.ones(len(risk))))
 assert page.shape==(len(risk),10) and np.isfinite(page).all() and np.isfinite(H).all()
 return dict(page=page,H=H,additive=additive,risk=risk,mass=m)
def initial_mask(p,own,K):
 order=np.argsort(-p['risk'],kind='stable');x=np.zeros(len(order),bool);x[own]=True;order=order[order!=own];x[order[:K-1]]=True;return x
def pool(p,x,own):
 r=np.flatnonzero(x);r=r[r!=own];a=np.flatnonzero(~x);r=r[np.argsort(p['risk'][r],kind='stable')[:16]];a=a[np.argsort(-p['risk'][a],kind='stable')[:32]]
 return np.repeat(r,len(a)),np.tile(a,len(r))
def state(p,x):return p['additive'][x].sum(0),p['H'][x].sum(0)
def features(p,x,own,st=None):
 r,a=pool(p,x,own);s,H=state(p,x) if st is None else st;count=s[5];scalar=np.array([s[0],s[1],max(0,s[3]/count-(s[2]/count)**2),s[4]/count]);hr=p['H'][r];ha=p['H'][a];hn=np.linalg.norm(H);rn=np.linalg.norm(hr,axis=1);an=np.linalg.norm(ha,axis=1);dr=hr@H;da=ha@H;change=ha-hr
 directional=np.column_stack((np.full(len(r),hn),dr,da,dr/(hn*rn+1e-30),da/(hn*an+1e-30),np.sum(hr*ha,axis=1)/(rn*an+1e-30),np.linalg.norm(H+change,axis=1)-hn,change@H/(np.linalg.norm(change,axis=1)*hn+1e-30)))
 F=np.column_stack((p['page'][r],p['page'][a],np.broadcast_to(scalar,(len(r),4)),directional));assert F.shape==(512,32) and np.isfinite(F).all();return F,r,a

def objective(raw,x):
 E=raw['e'][:,~x].sum(1);z=np.where(x[None],raw['exact_mass'],raw['summary_mass']).sum(1)
 return float(np.sum(np.square(E).sum(-1)/z**2)/(np.square(raw['dense_output']).sum()+1e-12))
def swap_labels(raw,x,r,a):
 E=raw['e'][:,~x].sum(1);z=np.where(x[None],raw['exact_mass'],raw['summary_mass']).sum(1);C=np.square(raw['dense_output']).sum()+1e-12;j=objective(raw,x);vals=[]
 for i in range(0,len(r),64):
  rr,aa=r[i:i+64],a[i:i+64];ee=E[:,None]+raw['e'][:,rr]-raw['e'][:,aa];zz=z[:,None]-raw['d'][:,rr]+raw['d'][:,aa];assert (zz>0).all();vals.append(np.sum(np.square(ee).sum(-1)/zz**2,axis=0)/C)
 return j-np.concatenate(vals)
def metric(g,j,oracle):
 rel=g/j;pos=oracle>1e-12
 return dict(n=len(g),mean_gain_pct=float(100*rel.mean()),median_gain_pct=float(100*np.median(rel)),p10_gain_pct=float(100*np.quantile(rel,.1)),improved=int(np.sum(g>1e-12)),degraded=int(np.sum(g<-1e-12)),degradation_rate=float(np.mean(g<-1e-12)),mean_pool_recovery=float(np.mean(g[pos]/oracle[pos])) if pos.any() else None,median_pool_recovery=float(np.median(g[pos]/oracle[pos])) if pos.any() else None,pooled_gain_recovery=float(g.sum()/oracle.sum()) if oracle.sum()>1e-12 else None,gate=bool(rel.mean()>0 and np.mean(g<-1e-12)<=.15))

def preregister():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():return
 dump(OUT/'protocol.json',dict(name='E12 Set-Conditional Swap Prediction',baseline='178f95a8 / summary_pretrained / G-KVR',manifest_sha256=sha(DATA/'manifest.json'),states=144,splits=dict(train=36,validation=36,evaluation=72),input_probes=16,evaluation_queries=64,pool=[16,32],initial='Risk own protected; fixed10%',no_op='fixed zero logit and zero true gain; swap only if max logit exceeds frozen nonnegative threshold',feature_names=FEATURE_NAMES,feature_slots=dict(A0=[0,20],A1=[0,24],A2=[0,32]),architecture='32 -> 32 GELU -> 1; identical1089 parameters; unused standardized feature slots zero',seed=17,second_seed=29,updates=2000,optimizer='Adam lr1e-3 betas(.9,.999) eps1e-8 weight_decay0 grad_clip1',batch='full batch of36 complete states; no independent-action sampling',loss='KL/soft-target cross entropy across512 swaps + no-op; gain normalized by per-state standard deviation; temperature0.3 fixed before training',label_scale='max(std(512 gains),1e-12), labels only; never inference feature',thresholds=[0,.1,.25,.5,1,2,4,8,16,'infinity'],selection='maximize mean relative gain subject to degradation<=15% on validation; no-op fallback allowed; evaluate threshold0 separately',second_seed_rule='if any primary model passes validation, retrain only highest validation gain passing architecture with seed29, same budget, select threshold on validation',multi_rule='only if selected architecture passes validation with BOTH seeds; freeze both models first, evaluate their own trajectories at1/2/4; no new training states in this first-round decision study',scope='three single-step architectures first; no global solving, residual predictors, video generation, M64 sweep or fused kernels; phase2 expanded-state training deferred',allowlist=list(ALLOW),created=time.time()))
