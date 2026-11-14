"""120 same-state events, two full 64-query tiles, all heads; no cross-partition masks."""
import math,copy,torch
from ..review_v1.ops import Summary,SummaryOps,RouteOps,AttentionOps,Config
from ..review_v1.ablation import mass_scores
from ..learned_cache.io import atomic
from .core import GROUPS,regroup

def join(a,b):return Summary(*(torch.cat((getattr(a,n),getattr(b,n)),1) for n in ('keys','mean','detail','stats')))

def dense(q,k,v):
 z=torch.bmm(q.float(),k.flatten(1,2).float().transpose(1,2))/math.sqrt(128)
 return torch.bmm(z.softmax(-1),v.flatten(1,2).float()),z.logsumexp(-1)

def metrics(a,z,ref,zref):
 e=(a-ref).square().sum();energy=ref.square().sum();dl=z-zref
 return dict(nmse=float(e/energy),squared_error=float(e),target_energy=float(energy),log_denominator_mae=float(dl.abs().mean()),denominator_relative_mae=float(torch.expm1(dl.double()).abs().mean()))

def panels(q,k,v,s,basis,masks):
 h,n,d=q.shape;p=k.shape[1];states={name:(q.new_zeros(h,n,d,dtype=torch.float32),q.new_full((h,n),-torch.inf,dtype=torch.float32)) for name in masks}
 for a in range(0,p,8):
  exact,lz,approx,ls=AttentionOps.pages(q,k[:,a:a+8],v[:,a:a+8],s.slice(a,a+8),basis)
  for name,mask in masks.items():
   select=mask[:,:,a:a+8];z=torch.where(select,lz,ls);value=torch.where(select[...,None],exact,approx)
   out,logz=states[name];cz=z.logsumexp(-1);nz=torch.logaddexp(logz,cz)
   states[name]=(out*(logz-nz).exp()[...,None]+torch.einsum('hqp,hqpd->hqd',(z-nz[...,None]).exp(),value),nz)
 return states

@torch.no_grad()
def evaluate(m,q,k,v,ids,tiles,path,identity):
 p=ids.shape[1];ref,zref=dense(q,k,v);current=SummaryOps.build(m,k[:,p:],v[:,p:],cheap=True);rows={}
 offsets=torch.linspace(0,63,16,device=q.device).round().long();probes=q.reshape(12,2,64,128).index_select(2,offsets).flatten(1,2);own=(tiles+p)[None].expand(12,-1)
 for group in GROUPS:
  kh,vh,newids,check=regroup(m,k[:,:p],v[:,:p],ids,group);kk=torch.cat((kh,k[:,p:]),1);vv=torch.cat((vh,v[:,p:]),1)
  target,zt=dense(q,kk,vv);check['dense_nmse']=float((target-ref).square().sum()/ref.square().sum());check['dense_max_abs']=float((target-ref).abs().max());assert check['dense_nmse']<2e-10 and check['dense_max_abs']<1e-3,check
  summary=join(SummaryOps.build(m,kh,vh),current);f,_=RouteOps.features(probes,summary)
  risk=(f*m.theta[:,None,None]).sum(-1).reshape(12,2,16,p+72).mean(2)
  mass=mass_scores(probes,summary,2,16);masks={};selections={}
  ownmask=torch.zeros(12,2,p+72,device=q.device,dtype=torch.bool).scatter_(-1,own[...,None],True)
  masks['representation']=ownmask.repeat_interleave(64,1)
  for route,score in [('mass',mass),('risk',risk)]:
   sel=RouteOps.select(score,own,Config());assert bool((sel['counts']==math.floor((p+72)*.1)).all());masks[route]=sel['mask'].repeat_interleave(64,1);selections[route]=sel['ids'].cpu()
  out=panels(q,kk,vv,summary,m.basis,masks);row={n:metrics(a,z,ref,zref) for n,(a,z) in out.items()}
  # Auxiliary pool-mismatch diagnostic only; never used in the 4x2 main matrix.
  uniform=copy.deepcopy(m)
  for n in ['pool_k','pool_v','pool_feature']:getattr(uniform,n).zero_()
  us=join(SummaryOps.build(uniform,kh,vh),current)
  u,uz=AttentionOps.evaluate(q,kk,vv,us,m.basis,masks['representation'])
  row['uniform_pool_diagnostic']=metrics(u,uz,ref,zref)
  row['identity']=check;row['exact_pages_per_tile_head']=math.floor((p+72)*.1);row['representation_exact_pages']=1
  row['variance']={n:float((x.float()-x.float().mean(2,keepdim=True)).square().mean()) for n,x in [('key_page',kh),('value_page',vh)]}
  rows[group]=row
 atomic(path,dict(**identity,tiles=tiles.tolist(),queries_per_tile=64,heads=12,rows=rows))
 return rows

class Collector:
 def __init__(self,model,root,case,seed):self.model=model;self.root=root;self.case=case;self.seed=seed;self.engine=None
 def __call__(self,l,q,kh,vh,store,start):
  c=self.engine.current['chunk'];step=self.engine.current['step']
  if c not in (2,5,8) or step not in (0,2) or l not in (1,8,15,22,29):return
  path=self.root/'local'/f'{self.case}_s{self.seed}'/f'c{c}_l{l}_s{step}.json'
  if path.exists():return
  t0=(17*l+13*c+7*step+self.seed)%72;tiles=torch.tensor([t0,(t0+36)%72],device=q.device)
  qs=q[0].transpose(0,1).reshape(12,72,64,128).index_select(1,tiles).flatten(1,2)
  k=kh[0].transpose(0,1).reshape(12,-1,64,128);v=vh[0].transpose(0,1).reshape_as(k)
  identity=dict(case=self.case,seed=self.seed,chunk=c,layer=l,step=step,source='G-KVR L3 178f95a8 frozen trajectory',history_pages=store.n)
  evaluate(self.model[l],qs,k,v,store.token_ids[:,:store.n],tiles,path,identity)
  print('LOCAL',identity,flush=True)
