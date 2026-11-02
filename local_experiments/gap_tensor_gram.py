"""Split-K Tensor Core Gram for large groups; fused iterations for small groups.
Hard sorting is retained. Changed FP order requires real quality checks.
"""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .gap_group_fused import scores
@tr.jit
def _gram(X,O,N:tl.constexpr,S:tl.constexpr,BK:tl.constexpr=128):
    h=tl.program_id(0);part=tl.program_id(1)
    t=part*BK+tl.arange(0,BK);d=tl.arange(0,32)
    x=tl.load(X+(h*N+t[:,None])*32+d[None,:],t[:,None]<N,0)
    g=tl.dot(tl.trans(x),x,input_precision='tf32x3')
    tl.store(O+(h*S+part)*1024+d[:,None]*32+d[None,:],g)
@tr.jit
def _direction(G,O):
    h=tl.program_id(0);d=tl.arange(0,32)
    g=tl.load(G+h*1024+d[:,None]*32+d[None,:])
    diag=tl.sum(tl.where(d[:,None]==d[None,:],g,0.),1)
    axis=tl.argmax(diag,0,tie_break_left=True);v=(d==axis).to(tl.float32)
    for _ in range(4):
        v=tl.sum(g*v[None,:],1)
        norm=tl.maximum(lib.sqrt_rn(tl.sum(v*v,0)),1.e-8);v=tl.div_rn(v,norm)
    tl.store(O+h*32+d,v)
def gram_scores(x):
    h,n,d=x.shape;xc=x-x.mean(1,keepdim=True);s=tr.cdiv(n,128)
    partial=torch.empty(h,s,32,32,device=x.device)
    _gram[(h,s)](xc,partial,n,s,128,num_warps=4,num_stages=1)
    gram=partial.sum(1);direction=torch.empty(h,32,device=x.device)
    _direction[(h,)](gram,direction,num_warps=4,enable_fp_fusion=False)
    return torch.bmm(xc,direction.unsqueeze(-1)).squeeze(-1)
def balanced_tensor_gram(x,factors):
    h,n,d=x.shape;assert d==32
    ids=torch.arange(n,device=x.device).expand(h,n).clone();groups=1
    for index,factor in enumerate(factors):
        size=n//groups;x=x.reshape(h*groups,size,d);ids=ids.reshape(h*groups,size)
        projection=scores(x) if size<=512 else gram_scores(x)
        order=projection.argsort(dim=-1,stable=True)
        if index+1<len(factors):x=x.gather(1,order[:,:,None].expand_as(x))
        ids=ids.gather(1,order);groups*=factor
    return ids.reshape(h,n)
