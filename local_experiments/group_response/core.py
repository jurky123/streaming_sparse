"""Grouping-only intervention. Pool features always remain the baseline definition."""
import math,torch,hashlib
from ..review_v1.ops import Config,SummaryOps
from ..response_hard import balanced
from ..cache_grouping import reorder
from .baseline_runtime import AblationRuntime
GROUPS=('G-K','G-VR','G-KVR','G-ORDER')

def responses(m,k,v,need_value=True,need_key=True):
 kc=k-k.mean(-2,keepdim=True);score=torch.einsum('hptd,hrd->hptr',kc,m.bank)/math.sqrt(128)
 key=score/score.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6) if need_key else None
 if not need_value:return key,None
 a=score.softmax(-2);vp=torch.einsum('hptd,hdr->hptr',v,m.group_v);mean=(a*vp).sum(-2,keepdim=True)
 vr=a*k.shape[-2]*(vp-mean);vr=vr/vr.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
 return key,vr

@torch.no_grad()
def grouping(m,k,v,group,config=Config()):
 # k/v: H,4608,128 in original token/RoPE order.
 assert k.shape==v.shape==(12,4608,128)
 if group=='G-ORDER':order=torch.arange(4608,device=k.device)[None].expand(12,-1)
 else:
  if group=='G-KVR':f=m.group_features(k.float()[:,None],v.float()[:,None]).squeeze(1)
  else:
   key,vr=responses(m,k.float()[:,None],v.float()[:,None],need_value=group=='G-VR',need_key=group=='G-K');f=(key if group=='G-K' else vr).squeeze(1)
  order=balanced(f,[3,3,2,2,2]);fp=reorder(f,order)
  inner=balanced(fp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
  order=order.reshape(12,72,64).gather(2,inner)[:,config.page_order(k.device)].flatten(1)
 return reorder(k,order).reshape(12,72,64,128),reorder(v,order).reshape(12,72,64,128),order

class Runtime(AblationRuntime):
 def __init__(self,engine,model,group,level,event_callback=None):
  self.group_id=group;self.level=level;self.event_callback=event_callback;self.timings=[];self.identity_checks=[]
  original=torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cuda',weights_only=False)['model']
  centers={l:original[f'layers.{l}.center'] for l in range(1,30)}
  super().__init__(engine,model,'no_downstream',centers,config=Config())
 def forward(self,l,*args):
  self.sparse=self.level=='L3' or (self.level=='L2' and l==15)
  return super().forward(l,*args)
 def group(self,l,k,v):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  out=grouping(self.model[l],k[0].transpose(0,1),v[0].transpose(0,1),self.group_id,self.config);b.record()
  self.timings.append(('group',l,self.engine.current['chunk'],a,b))
  order=out[2];assert torch.equal(order.sort(-1).values,torch.arange(4608,device=k.device)[None].expand(12,-1))
  assert torch.equal(out[0].flatten(1,2),reorder(k[0].transpose(0,1),order)) and torch.equal(out[1].flatten(1,2),reorder(v[0].transpose(0,1),order))
  self.identity_checks.append(dict(layer=l,chunk=self.engine.current['chunk'],bijective=True,kv_matches_token_identity=True))
  return out
 def commit_summary(self,l,store,k,v,order,start):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);c=torch.cuda.Event(enable_timing=True);a.record()
  summary=SummaryOps.build(self.model[l],k,v);b.record();store.commit(k,v,summary,order,start);c.record()
  self.timings.extend([('summary',l,self.engine.current['chunk'],a,b),('write',l,self.engine.current['chunk'],b,c)])
 def diagnostics(self):
  torch.cuda.synchronize()
  return dict(costs=[dict(kind=n,layer=l,chunk=c,milliseconds=a.elapsed_time(b)) for n,l,c,a,b in self.timings],identity_checks=self.identity_checks,group=self.group_id,level=self.level)

@torch.no_grad()
def regroup(m,k,v,ids,group):
 """Replace complete chunks in-place in page space, never change survivors."""
 h,p,t,d=k.shape;ko=k.clone();vo=v.clone();io=ids.clone();full=[];partial=[]
 chunks=torch.unique(ids[0]//4608).tolist()
 for chunk in chunks:
  slots=torch.where(ids[0,:,0]//4608==chunk)[0]
  assert bool((ids[:,slots]//4608==chunk).all())
  if slots.numel()!=72:partial.extend(slots.tolist());continue
  orig=ids[:,slots].flatten(1);sort=orig.argsort(-1);kk=reorder(k[:,slots].flatten(1,2),sort);vv=reorder(v[:,slots].flatten(1,2),sort)
  assert torch.equal(orig.gather(1,sort),torch.arange(chunk*4608,(chunk+1)*4608,device=k.device)[None].expand(12,-1))
  kg,vg,order=grouping(m,kk,vv,group)
  ko[:,slots]=kg;vo[:,slots]=vg;io[:,slots]=(order+chunk*4608).reshape(12,72,64);full.append(chunk)
 assert torch.equal(ids.flatten(1).sort(-1).values,io.flatten(1).sort(-1).values)
 assert torch.equal(ko[:,partial],k[:,partial]) and torch.equal(vo[:,partial],v[:,partial]) and torch.equal(io[:,partial],ids[:,partial])
 oldorder=ids.flatten(1).argsort(-1);neworder=io.flatten(1).argsort(-1)
 assert torch.equal(reorder(k.flatten(1,2),oldorder),reorder(ko.flatten(1,2),neworder))
 assert torch.equal(reorder(v.flatten(1,2),oldorder),reorder(vo.flatten(1,2),neworder))
 identity_sha=hashlib.sha256(ids.flatten(1).sort(-1).values.cpu().numpy().tobytes()).hexdigest()
 return ko,vo,io,dict(sorted_token_ids_sha256=identity_sha,unique_tokens_per_head=[int(torch.unique(x).numel()) for x in io.flatten(1)],full_chunks=full,partial_pages=len(partial),regrouped_pages=72*len(full),history_pages=p,regroup_fraction=72*len(full)/p,token_bijection=True,token_kv_exact=True,partial_unchanged=True)
