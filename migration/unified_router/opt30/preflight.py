"""New algorithm tests on real replay: mean initialization, independent pooling,
permutation bijection, native forward/loss equivalence, and usable gradients."""
import json,gc,torch
import torch.nn.functional as F
from .setup import ROOT,VARIANTS
from .data import Events
from .model import QueryRouter
from . import kernels
from ..opt16.grouping import order,labels
from ..common import atomic_json,paired_seed
from ..opt29.preflight import verify_kernel

@torch.no_grad()
def pool_reference(q,w,g,r):
 h,n,d=q.shape;t=(n+g-1)//g
 z=F.pad(q.float(),(0,0,0,t*g-n)).reshape(h,t,g,d);valid=torch.arange(t*g,device=q.device).reshape(t,g)<n
 zn=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)
 logit=torch.einsum('htgd,hrd->htgr',zn,w.float())/8
 anchor=F.one_hot(torch.arange(g,device=q.device)//(g//r),r).bool()[None,None]&valid[None,:,:,None]
 a=torch.exp(2*(logit/2).tanh())*anchor
 bar=torch.einsum('htgr,htgd->htrd',a,z)/a.sum(2).clamp_min(1e-20)[...,None]
 mass=anchor.float().sum(2)/valid.sum(-1)[None,:,None]
 return bar,mass

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;paired_seed(30024)
 m=json.loads((ROOT/'experiment.json').read_text());events=Events(m,24);q,raw,truth,targets=events.gpu(events.train[0]);rows=[]
 for name in VARIANTS:
  model=QueryRouter(m,24,name).cuda();model.initialize_groups()
  actual,d=model.deployed(q,raw,targets);assert torch.isfinite(actual).all()and float(d['executed'])<.22
  for side,x in [('q',q),('k',raw['kc'])]:
   ix=order(x,model.tree(side));reference=torch.argsort(labels(x,model.tree(side)),stable=True)
   assert torch.equal(ix.long(),reference)
  fine=verify_kernel(d,model.options);zero_exact=None;pool_error=None
  if model.options['mode']=='local':
   model.options['mode']='mean';mean,_=model.deployed(q,raw,targets);model.options['mode']='local';assert torch.equal(actual,mean),'Local initialization must exactly equal mean pooling';zero_exact=True
   with torch.no_grad():
    model.query_weight.normal_(std=.2)
    b,mass=kernels.local_aggregate(d['projected'],model.query_weight,model.options['g'],model.options['r']);ref,rm=pool_reference(d['projected'],model.query_weight,model.options['g'],model.options['r'])
    pool_error=float((b.float()-ref).square().sum()/ref.square().sum());assert pool_error<1e-5;assert torch.allclose(mass,rm.expand_as(mass),atol=1e-7,rtol=0)
    model.query_weight.zero_()
  output,info=model(q,raw,targets);assert torch.equal(actual,output.detach())
  loss=(output-truth).square().sum()/truth.square().sum();loss.backward()
  assert all(p.grad is None or torch.isfinite(p.grad).all()for p in model.trainable())
  gradient=float(model.query_weight.grad.norm())if model.query_weight.grad is not None else None
  if model.options['mode']=='local':assert gradient>0
  opt=torch.optim.AdamW(model.trainable(),lr=1e-5);opt.step();assert all(torch.isfinite(p).all()for p in model.trainable())
  row=dict(variant=name,actual_nmse=float(loss.detach()),native_equals_loss=True,mean_initialization_exact=zero_exact,pool_reference_nmse=pool_error,query_gradient=gradient,fine_max_abs=fine,executed=float(d['executed']))
  rows.append(row);print(row,flush=True);atomic_json(ROOT/'preflight.json',dict(passed=False,rows=rows))
  del model,opt,actual,d,output,loss;gc.collect();torch.cuda.empty_cache()
 atomic_json(ROOT/'preflight.json',dict(passed=True,rows=rows,scope='Real Train replay at L24; all variants; independent FP32 pooling reference and native forward agreement. Not final quality.'))
if __name__=='__main__':main()
