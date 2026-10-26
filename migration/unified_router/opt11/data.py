import torch
from pathlib import Path
from ..common import load_raw,event_key
from ..core import build
from ..fast_route import prepare
from ..opt9.select import fused_select
from ..opt10.compact_route import CompactMomentRoute
from ..learning import ActionHead
from ..opt9.segmented_attention import attention as compact_attention
from .attention import attention as aggressive_attention
from .head import actions,apply,exact,subset

VARIANTS=['Compact','Aggressive']
def kernel(variant):return compact_attention if variant=='Compact'else aggressive_attention
def attend(variant,q,raw,mask):return kernel(variant)(q,raw['kh'],raw['vh'],raw['kc'],raw['vc'],mask,pack_v=True,fuse_pack=True)
def initial_head(base,layer):
 state=torch.load(Path(base)/f'P3_training/layer{layer:02d}/K32-Set/seed0/best.pt',map_location='cpu',weights_only=True)['head'];head=ActionHead(state).cuda();head.load_state_dict(state);return head
class Builder:
 def __init__(self,base,layer,beta):
  state=torch.load(Path(base)/f'action_basis/layer{layer:02d}.pt',map_location='cpu',weights_only=True);self.p={k:v.cuda()for k,v in state['parameters'].items()};self.route=CompactMomentRoute(batch=4680);self.build=torch.compile(build,fullgraph=True,dynamic=False);self.prepare=torch.compile(prepare,fullgraph=True,dynamic=False);self.beta=beta
 @torch.no_grad()
 def inputs(self,q,raw,step):
  sketch=self.build(raw['kh'],raw['vh'],raw['kc'],raw['vc'],self.p,need_value=False);prepared=self.prepare(sketch,'K32-Set');state=self.route.begin(prepared,sketch);score,weights=self.route.score(q,state);mask,rem,add=fused_select(score,sketch.origins,self.beta,mode='packed32');stats,cv=self.route.stats(q,sketch,score,mask,rem,add,weights,state)
  h,n,_=score.shape;stats=stats.reshape(h,n,80);stats[...,73:]=0.
  d=dict(stats=stats,cv=cv.reshape(h,n,16,32),score=score,base_mask=mask,rem=rem.reshape(h,n,8),add=add.reshape(h,n,8),costs=sketch.costs,origins=sketch.origins,history_tokens=sketch.history_tokens,step=step,beta=self.beta)
  return d,(sketch,prepared,state)

def load_group(root,row,layer):
 f=Path(root)/'group_queries'/event_key(row)/f'layer{layer:02d}.pt';d=torch.load(f,map_location='cpu',weights_only=True,mmap=True);assert d['sample_id']==row['sample_id']and d['indices']==row['group_query_indices'];return d['q'].cuda()
def feature_path(root,row,layer):return Path(root)/'features'/f'layer{layer:02d}'/(event_key(row)+'.pt')
def device(d):return {k:v.cuda()if isinstance(v,torch.Tensor)else v for k,v in d.items()}
def cpu(d):return {k:v.detach().cpu().contiguous().clone()if isinstance(v,torch.Tensor)else v for k,v in d.items()}
@torch.no_grad()
def dense(q,raw):return torch.nn.functional.scaled_dot_product_attention(q[None],torch.cat((raw['kh'],raw['kc']),1)[None],torch.cat((raw['vh'],raw['vc']),1)[None])[0]
@torch.no_grad()
def labels(head,d,q,raw,truth,targets):
 base=apply(d,exact(head,d).argmax(-1));target=torch.tensor(targets,device=q.device);small=subset(d,target);a=actions(small);page=torch.arange(d['score'].shape[-1],device=q.device);result={v:torch.empty((q.shape[0],len(targets),81),device=q.device)for v in VARIANTS};mask=base.clone()
 for ai in range(81):
  candidate=(small['base_mask']&(page!=a['remove'][...,ai,None]))|(page==a['add'][...,ai,None]);candidate=torch.where(a['valid'][...,ai,None],candidate,small['base_mask']);mask[:,target]=candidate
  for variant in VARIANTS:
   y=attend(variant,q,raw,mask);result[variant][...,ai]=(y[:,target].float()-truth[:,target].float()).square().sum(-1)
 assert all(torch.isfinite(r).all()for r in result.values())
 return result,truth[:,target].float().square().sum(),base
