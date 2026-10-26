"""Actual native forward, explicitly biased page-gate backward for joint routing.
No gradients through discrete top-k or the CUDA attention kernel are claimed.
"""
import math,torch
import torch.nn.functional as F
from .setup import PARENT,PREVIOUS,ROOT,VARIANTS
from ..opt17.model import JointGroupedRouter
from ..opt16.pipeline import GroupedRouter
from ..opt16.grouping import gather
from ..opt12.precision import HalfBmmFP32Backward,fp16_round,align
from ..opt12.model import ActualAttention
from ..core import build
from .pipeline import Pipeline
@torch.no_grad()
def page_statistics(q,kv,seg,k):
 zs=[];values=[];h,t,d=q.shape
 for start in range(0,kv.k.shape[1],seg):
  kk=kv.k[:,start:start+seg];vv=kv.v[:,start:start+seg];b=math.ceil(seg/k)
  ll=torch.bmm(q.float(),kk.float().transpose(1,2))/math.sqrt(d);ll=F.pad(ll,(0,b*k-seg),value=-torch.inf).reshape(h,t,b,k)
  vp=F.pad(vv.float(),(0,0,0,b*k-seg)).reshape(h,b,k,d)
  zs.append(ll.logsumexp(-1));values.append(torch.einsum('htbk,hbkd->htbd',ll.softmax(-1),vp))
 return torch.cat(zs,2),torch.cat(values,2)
class QueryRouter(JointGroupedRouter):
 def __init__(self,m,layer,variant):
  super().__init__(m,layer,'Tile20')
  origin='q32_full' if VARIANTS[variant]['g']==32 else 'q64_mean8'
  state=torch.load(PREVIOUS/f'training/layer{layer:02d}/{origin}/best.pt',map_location='cpu',weights_only=True)
  self.load_state_dict({k:v for k,v in state['model'].items()if not k.startswith('query_')})
  self.variant=variant;self.options=dict(VARIANTS[variant]);self.budget=.2;o=self.options
  self.query_weight=torch.nn.Parameter(torch.zeros(12,o['r'],64),requires_grad=o['mode']=='local')
  self.query_bias=torch.nn.Parameter(torch.zeros(12,o['r']),requires_grad=False)
 def initialize_groups(self):
  side=self.options['grouping']
  if side=='none':return
  trees=torch.load(ROOT/f'grouping/layer{self.layer:02d}/trees.pt',map_location=self.query_weight.device,weights_only=True)
  with torch.no_grad():
   prefix='raw_'if side=='rawqk'else'joint_'if side=='qk'else''
   for s in ('qk'if side=='rawqk'else side):
    getattr(self,s+'_tree_weight').copy_(trees[prefix+s+'_weight']);getattr(self,s+'_tree_threshold').copy_(trees[prefix+s+'_threshold'])
 def trainable(self):return [p for p in self.parameters()if p.requires_grad]
 @torch.no_grad()
 def deployed(self,q,raw,targets,previous_q=None,**unused):
  self.pipeline=GroupedRouter(self.dictionary(),self.tree('q'),self.tree('k'),compile=False)
  runner=Pipeline(self,raw);full,d=runner.run(q,raw['kc'],raw['vc'],True,previous_q=previous_q);ix=d['q_order'];h,n,_=q.shape;o=self.options;g=o['g'];k=o['k'];b=runner.lib.b
  assert bool(((d['selection']>=0)&(d['selection']<b)).all())and bool((d['selection'].diff(dim=-1)>0).all())
  inv=torch.empty_like(ix);inv.scatter_(1,ix.long(),torch.arange(n,device=q.device,dtype=ix.dtype)[None].expand_as(ix));pos=inv[:,targets].long();group=pos//g;head=torch.arange(h,device=q.device)[:,None]
  d.update(group=group,selected=d['selection'][head,group].long(),target_candidates=d['candidates'][head,group].long())
  d['mask']=torch.zeros((h,len(targets),b),device=q.device).scatter_(-1,d['selected'],1.)
  qcost=(n-torch.arange(d['selection'].shape[1],device=q.device)*g).clamp(max=g).float();length=(n-(d['selection']%runner.lib.pages)*k).clamp(max=k).float()
  d['keep']=(length.sum(-1)*qcost).sum()/(h*n*d['kv'].k.shape[1]);d['executed']=q.new_tensor(d['selection'].numel()*g*k/(h*n*d['kv'].k.shape[1]),dtype=torch.float32)
  return full[:,targets].float(),d
 def scores_z(self,z,sketch):
  basis=sketch.key_basis[...,:64];out=[]
  for part in sketch.parts:
   h,b,p,d=part.key.shape;keys=torch.bmm(part.key.flatten(1,2),basis)
   ll=HalfBmmFP32Backward.apply(z,keys.transpose(1,2)).reshape(h,z.shape[1],b,p)/math.sqrt(128)
   lc=part.count.clamp_min(1e-30).log().masked_fill(part.count<=1e-8,-torch.inf)
   out.append((ll+lc[:,None]).logsumexp(-1))
  return torch.cat(out,2)
 def forward(self,q,raw,targets,previous_q=None,temperature=.5):
  o=self.options;g,r,k=o['g'],o['r'],o['k'];h,n,_=q.shape;t=len(targets);head=torch.arange(h,device=q.device)[:,None]
  with torch.no_grad():
   actual,d=self.deployed(q,raw,targets,previous_q);kv=d['kv'];nh=kv.history_tokens;logz,value=page_statistics(q[:,targets],kv,n,k)
   pos=d['group'][...,None]*g+torch.arange(g,device=q.device);valid=pos<n;safe=pos.clamp_max(n-1)
   original=gather(d['q_order'][...,None],safe.flatten(1)).squeeze(-1).long()
   members=gather(q,original).float();source=gather(previous_q,original).float()if previous_q is not None else members
   actual_bar=d['bar'][head,d['group']];actual_mass=d['mass'][head,d['group']];actual_fine=d['fine'][head,d['group']]
  basis=self.routing['key_projection'];z=fp16_round(HalfBmmFP32Backward.apply(members,basis)).reshape(h,t,g,64)
  if o['mode']=='local':
   # Fixed non-overlapping coverage; only pooling weights inside each subgroup learn.
   # RMS-normalized scoring, bounded ratio exp(4), exact mean at zero initialization.
   zn=z.float()*torch.rsqrt(z.float().square().mean(-1,keepdim=True)+1e-6)
   logits=HalfBmmFP32Backward.apply(zn.flatten(1,2),self.query_weight.transpose(1,2)).reshape(h,t,g,r)/8
   logits=2*torch.tanh(logits/2)
   anchor=F.one_hot(torch.arange(g,device=q.device)//(g//r),r).bool()[None,None]
   active=anchor&valid[...,None]
   a=fp16_round(torch.exp(logits)*active)
  elif o['mode']=='mean':a=F.one_hot(torch.arange(g,device=q.device)//(g//r),r).float()[None,None]*valid[...,None]
  elif o['mode']=='sample':
   positions=(torch.arange(r,device=q.device)*(g//r)+(g//r)//2)[None,None].expand(h,t,-1);positions=torch.minimum(positions,valid.sum(-1)[...,None]-1);bar=z.gather(2,positions[...,None].expand(-1,-1,-1,64));mass=torch.full((h,t,r),1/r,device=q.device)
  else:bar=z;mass=valid.float()/valid.sum(-1,keepdim=True)
  if o['mode']in ['local','mean']:
   count=a.sum(2);bar=torch.einsum('htgr,htgd->htrd',a,z)/count.clamp_min(1e-20)[...,None]
   coverage=F.one_hot(torch.arange(g,device=q.device)//(g//r),r).float()[None,None]*valid[...,None]
   mass=coverage.sum(2)/valid.sum(-1,keepdim=True)
  bar=align(bar,actual_bar.float());mass=align(mass,actual_mass)
  grouped=dict(kh=kv.k[:,:nh],vh=kv.v[:,:nh],kc=kv.k[:,nh:],vc=kv.v[:,nh:]);sketch=build(**grouped,parameters=self.dictionary(),need_value=False)
  score=self.scores_z(bar.flatten(1,2),sketch).reshape(h,t,r,-1);b64=score.shape[-1];cid=d['target_candidates'][:,:,None].expand(-1,-1,r,-1)
  score=align(score,score.detach().clone().scatter_(-1,cid,actual_fine))
  allowed=torch.zeros((h,t,b64),device=q.device,dtype=torch.bool).scatter_(-1,d['target_candidates'],True)
  prob=score.masked_fill(~allowed[:,:,None],-torch.inf).softmax(-1);importance=(prob*mass[...,None]).sum(2)
  if k==128:
   p64=math.ceil(n/64);pk=math.ceil(n/128);parts=importance.reshape(h,t,-1,p64);importance=F.pad(parts,(0,pk*2-p64)).reshape(h,t,-1,pk,2).sum(-1).flatten(2)
  hard_importance=d['importance'][head,d['group']];importance=align(importance,hard_importance)
  boundary=d['selected'].gather(-1,hard_importance.gather(-1,d['selected']).argmin(-1,keepdim=True))
  logits=(importance+1e-6).log();soft=((logits-logits.gather(-1,boundary))/temperature).sigmoid();gate=align(soft,d['mask'])
  output=ActualAttention.apply(gate,actual,logz,value)
  return output,dict(keep=d['keep'],executed=d['executed'],assignment_entropy=-(mass*mass.clamp_min(1e-20).log()).sum(-1).mean())
