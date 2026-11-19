"""120 same-state events, two full 64-query tiles, all heads; no cross-partition masks."""
import math,copy,torch,json,hashlib
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
 errors={name:dict(total=0.,history=0.,tail=0.,tail_count=0) for name in masks}
 for a in range(0,p,8):
  exact,lz,approx,ls=AttentionOps.pages(q,k[:,a:a+8],v[:,a:a+8],s.slice(a,a+8),basis)
  for name,mask in masks.items():
   select=mask[:,:,a:a+8];err=(ls-lz).abs().masked_fill(select,0);errors[name]['total']+=float(err.double().sum());errors[name]['history']+=float(err[:,:,:max(0,min(8,p-72-a))].double().sum());errors[name]['tail_count']+=int((~select).sum());z=torch.where(select,lz,ls);value=torch.where(select[...,None],exact,approx)
   out,logz=states[name];cz=z.logsumexp(-1);nz=torch.logaddexp(logz,cz)
   states[name]=(out*(logz-nz).exp()[...,None]+torch.einsum('hqp,hqpd->hqd',(z-nz[...,None]).exp(),value),nz)
 return states,{name:dict(page_log_denominator_mae=e['total']/(h*n*p),history_page_log_denominator_mae=e['history']/(h*n*(p-72)),unselected_page_log_denominator_mae=e['total']/e['tail_count']) for name,e in errors.items()}

@torch.no_grad()
def evaluate(m,q,k,v,ids,tiles,path,identity,groups=GROUPS):
 p=ids.shape[1];ref,zref=dense(q,k,v);current=SummaryOps.build(m,k[:,p:],v[:,p:],cheap=True);rows={}
 offsets=torch.linspace(0,63,16,device=q.device).round().long();probes=q.reshape(12,2,64,128).index_select(2,offsets).flatten(1,2);own=(tiles+p)[None].expand(12,-1)
 for group in groups:
  kh,vh,newids,check=regroup(m,k[:,:p],v[:,:p],ids,group);kk=torch.cat((kh,k[:,p:]),1);vv=torch.cat((vh,v[:,p:]),1)
  target,zt=dense(q,kk,vv);check['dense_nmse']=float((target-ref).square().sum()/ref.square().sum());check['dense_max_abs']=float((target-ref).abs().max());assert check['dense_nmse']<2e-10 and check['dense_max_abs']<1e-3,check
  summary=join(SummaryOps.build(m,kh,vh),current);f,_=RouteOps.features(probes,summary)
  risk=(f*m.theta[:,None,None]).sum(-1).reshape(12,2,16,p+72).mean(2)
  mass=mass_scores(probes,summary,2,16);masks={};selections={}
  ownmask=torch.zeros(12,2,p+72,device=q.device,dtype=torch.bool).scatter_(-1,own[...,None],True)
  masks['representation']=ownmask.repeat_interleave(64,1)
  for route,score in [('mass',mass),('risk',risk)]:
   sel=RouteOps.select(score,own,Config());assert bool((sel['counts']==math.floor((p+72)*.1)).all());masks[route]=sel['mask'].repeat_interleave(64,1);selections[route]=sel['ids'].cpu()
  out,page_errors=panels(q,kk,vv,summary,m.basis,masks);row={n:metrics(a,z,ref,zref)|page_errors[n] for n,(a,z) in out.items()}
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
  path=self.root/'restored_controls'/f'{self.case}_s{self.seed}'/f'c{c}_l{l}_s{step}.json'
  if path.exists():return
  t0=(17*l+13*c+7*step+self.seed)%72;tiles=torch.tensor([t0,(t0+36)%72],device=q.device)
  qs=q[0].transpose(0,1).reshape(12,72,64,128).index_select(1,tiles).flatten(1,2)
  k=kh[0].transpose(0,1).reshape(12,-1,64,128);v=vh[0].transpose(0,1).reshape_as(k);ids=store.token_ids[:,:store.n]
  snapshot=self.root/'qkv'/f'{self.case}_s{self.seed}'/f'c{c}_l{l}_s{step}.pt'
  atomic(snapshot,dict(q=qs.cpu(),k=k.cpu(),v=v.cpu(),ids=ids.cpu(),tiles=tiles.cpu()),True)
  identity=dict(case=self.case,seed=self.seed,chunk=c,layer=l,step=step,source='deterministic replay of preceding G-KVR L3 trajectory',history_pages=store.n,qkv_path=str(snapshot),qkv_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
  rows=evaluate(self.model[l],qs,k,v,ids,tiles,path,identity,groups=['G-KVR','G-ORDER'])
  old=json.loads((self.root.parent/'group_response_results/local'/f'{self.case}_s{self.seed}'/path.name).read_text());checks=[]
  for group in ['G-ORDER','G-KVR']:
   for panel in ['representation','mass','risk']:
    for metric in ['nmse','squared_error','target_energy','log_denominator_mae']:
     a=rows[group][panel][metric];b=old['rows'][group][panel][metric];error=abs(a-b);assert error<=1e-6*max(abs(b),1e-8),(group,panel,metric,a,b)
     checks.append(dict(group=group,panel=panel,metric=metric,abs_delta=error))
  atomic(path.with_suffix('.reuse'),dict(passed=True,checks=checks));print('RESTORED',identity,flush=True)
