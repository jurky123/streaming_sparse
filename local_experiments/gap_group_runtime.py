"""Native-order normalization fusion in hard balanced grouping; inference only."""
import torch
from .gap_group_norm import normalize
from .gap_commit_graph_runtime import GapCommitGraphRuntime
from . import gap_commit_graph_runtime as commit_module

def balanced(x,factors):
    h,n,d=x.shape;assert d==32
    ids=torch.arange(n,device=x.device).expand(h,n).clone();groups=1
    for index,factor in enumerate(factors):
        size=n//groups;x=x.reshape(h*groups,size,d);ids=ids.reshape(h*groups,size)
        xc=x-x.mean(1,keepdim=True);var=xc.square().mean(1)
        direction=torch.zeros_like(var).scatter_(1,var.argmax(-1,keepdim=True),1.)
        for _ in range(4):
            direction=torch.bmm(xc.transpose(1,2),torch.bmm(xc,direction.unsqueeze(-1))).squeeze(-1)
            direction=normalize(direction)
        order=torch.bmm(xc,direction.unsqueeze(-1)).squeeze(-1).argsort(dim=-1,stable=True)
        if index+1<len(factors):x=x.gather(1,order[:,:,None].expand_as(x))
        ids=ids.gather(1,order);groups*=factor
    return ids.reshape(h,n)
class GapGroupRuntime(GapCommitGraphRuntime):
    def commit(self,*a,**kw):
        old=commit_module.balanced
        try:
            commit_module.balanced=balanced
            return super().commit(*a,**kw)
        finally:commit_module.balanced=old
    def finish(self):return super().finish()|dict(grouping='native-order fused norm32; no final unused feature gather')
