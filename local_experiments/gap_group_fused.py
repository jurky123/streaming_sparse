"""Fused small-group power iterations; hard, page-local grouping retained.
Changes floating point reductions; physical ordering must be quality validated.
"""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .gap_group_norm import normalize
@tr.jit
def _scores(X,O,N:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);t=tl.arange(0,B);d=tl.arange(0,32)
    x=tl.load(X+(group*N+t[:,None])*32+d[None,:],t[:,None]<N,0)
    mean=tl.sum(x,0)/N;x=tl.where(t[:,None]<N,x-mean[None,:],0.)
    var=tl.sum(x*x,0)/N;axis=tl.argmax(var,0,tie_break_left=True)
    direction=(d==axis).to(tl.float32)
    for _ in range(4):
        projection=tl.sum(x*direction[None,:],1)
        direction=tl.sum(x*projection[:,None],0)
        norm=tl.maximum(lib.sqrt_rn(tl.sum(direction*direction,0)),1.e-8)
        direction=tl.div_rn(direction,norm)
    projection=tl.sum(x*direction[None,:],1)
    tl.store(O+group*N+t,projection,t<N)
def scores(x):
    h,n,d=x.shape;assert d==32 and x.is_contiguous()
    out=torch.empty(h,n,device=x.device)
    _scores[(h,)](x,out,n,tr.next_power_of_2(n),num_warps=4,enable_fp_fusion=False)
    return out

def balanced_fused(x,factors,threshold=128):
    h,n,d=x.shape;assert d==32
    ids=torch.arange(n,device=x.device).expand(h,n).clone();groups=1
    for index,factor in enumerate(factors):
        size=n//groups;x=x.reshape(h*groups,size,d);ids=ids.reshape(h*groups,size)
        if size<=threshold:projection=scores(x)
        else:
            xc=x-x.mean(1,keepdim=True);var=xc.square().mean(1)
            axis=var.argmax(-1,keepdim=True)
            projection=xc.gather(2,axis[:,None,:].expand(-1,size,1))
            direction=normalize(torch.bmm(xc.transpose(1,2),projection).squeeze(-1))
            for _ in range(3):direction=normalize(torch.bmm(xc.transpose(1,2),torch.bmm(xc,direction.unsqueeze(-1))).squeeze(-1))
            projection=torch.bmm(xc,direction.unsqueeze(-1)).squeeze(-1)
        order=projection.argsort(dim=-1,stable=True)
        if index+1<len(factors):x=x.gather(1,order[:,:,None].expand_as(x))
        ids=ids.gather(1,order);groups*=factor
    return ids.reshape(h,n)
