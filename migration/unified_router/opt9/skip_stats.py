"""Instrumentation only: count PV tiles outside timed/profiler passes.
Online softmax block skipping probe, inspired by SpargeAttn / BLASST.
Independent per-query masks; CTA skips only when all visible queries have low logits.
Threshold uses base-2 logits, not natural logarithms.
"""
import torch
import triton as tr
import triton.language as tl
from ..fast_kernels import _union
from triton.language.extra.cuda.libdevice import nearbyint
@tr.jit
def _split(QP,KP,VP,KS,M,IDS,COUNTS,OUT,LSE,EXEC,
               Q:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
               QH:tl.constexpr,QQ:tl.constexpr,KH:tl.constexpr,KK:tl.constexpr,
               VH:tl.constexpr,VK:tl.constexpr,G:tl.constexpr,SPLITS:tl.constexpr,PVFP16:tl.constexpr, THRESH:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2)
    qi=g*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,64)
    q=tl.load(QP+h*QH+qi[:,None]*QQ+d[None,:],qi[:,None]<Q,0)
    qs=tl.maximum(tl.max(tl.max(tl.abs(q.to(tl.float32)),1),0)/127.,1.e-10)
    q=nearbyint(q.to(tl.float32)/qs).to(tl.int8)
    maximum=tl.full((G,),float('-inf'),tl.float32)
    denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
    count=tl.load(COUNTS+h*tr.cdiv(Q,G)+g)
    executed=0
    part=tl.cdiv(count,SPLITS)
    for j in range(split*part,tl.minimum((split+1)*part,count)):
        page=tl.load(IDS+(h*tr.cdiv(Q,G)+g)*B+j)
        # 4680-token segments have 73 full pages and one 8-token tail.
        segment=page//74;local=page%74;ki=segment*4680+local*64+t
        valid=(ki<K)&(local*64+t<4680)
        k=tl.load(KP+h*KH+ki[None,:]*KK+d[:,None],valid[None,:],0)
        ks=tl.load(KS+h*B+page)
        logits=tl.dot(q,k).to(tl.float32)*(qs*ks)*0.12751743082459617
        visible=tl.load(M+(h*Q+qi)*B+page,qi<Q,False)
        logits=tl.where(visible[:,None]&valid[None,:],logits,float('-inf'))
        page_max=tl.max(logits,1)
        gap=tl.where(page_max==float('-inf'),float('-inf'),page_max-maximum)
        if tl.max(gap,0)>=THRESH:
            executed+=1
            newmax=tl.maximum(maximum,page_max)
            safe=tl.where(newmax==float('-inf'),0.,newmax)
            alpha=tl.exp2(maximum-safe)
            prob=tl.exp2(logits-safe[:,None])
            v=tl.load(VP+h*VH+ki[:,None]*VK+d[None,:],valid[:,None],0)
            if PVFP16:v=v.to(tl.float16)
            acc=acc*alpha[:,None]+tl.dot(prob.to(v.dtype),v)
            denom=denom*alpha+tl.sum(prob,1);maximum=newmax
    tl.store(EXEC+(h*tr.cdiv(Q,G)+g)*SPLITS+split,executed)
    normal=tl.where(denom>0,denom,1.)
    tl.store(OUT+((h*Q+qi[:,None])*SPLITS+split)*128+d[None,:],acc/normal[:,None],qi[:,None]<Q)
    tl.store(LSE+(h*Q+qi)*SPLITS+split,maximum+tl.log2(normal),qi<Q)


@tr.jit
def _merge(P,L,O,Q:tl.constexpr,S:tl.constexpr):
    row=tl.program_id(0);s=tl.arange(0,S);d=tl.arange(0,128)
    lse=tl.load(L+row*S+s);maximum=tl.max(lse,0)
    weight=tl.exp2(lse-maximum);denom=tl.sum(weight,0)
    y=tl.load(P+(row*S+s[:,None])*128+d[None,:])
    out=tl.sum(y*weight[:,None],0)/denom
    tl.store(O+row*128+d,out)

@tr.jit
def _quantize(K,O,S,N:tl.constexpr,B:tl.constexpr,HS:tl.constexpr,NS:tl.constexpr):
    h=tl.program_id(1);page=tl.program_id(0);t=tl.arange(0,64);d=tl.arange(0,128)
    ki=page//74*4680+page%74*64+t;valid=(page%74*64+t<4680)&(ki<N)
    k=tl.load(K+h*HS+ki[:,None]*NS+d[None,:],valid[:,None],0.).to(tl.float32)
    scale=tl.maximum(tl.max(tl.max(tl.abs(k),1),0)/127.,1.e-10)
    quant=nearbyint(k/scale).to(tl.int8)
    tl.store(O+(h*N+ki[:,None])*128+d[None,:],quant,valid[:,None])
    tl.store(S+h*B+page,scale)

def attention(q,k,v,mask,group=32,splits=2,warps=4,stages=2,pvfp16=False,precast=False,return_union=False,threshold=-8.):
    h,n,d=q.shape;b=mask.shape[-1]
    if precast:v=v.half()
    ki=torch.empty_like(k,dtype=torch.int8,memory_format=torch.contiguous_format)
    ks=torch.empty((h,b),device=k.device,dtype=torch.float32)
    _quantize[(b,h)](k,ki,ks,k.shape[1],b,k.stride(0),k.stride(1),num_warps=4)
    ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32)
    counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
    _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
    lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
    executed=torch.empty((h,tr.cdiv(n,group),splits),device=q.device,dtype=torch.int32)
    _split[(tr.cdiv(n,group),h,splits)](q,ki,v,ks,mask,ids,counts,partial,lse,executed,n,k.shape[1],b,
        q.stride(0),q.stride(1),ki.stride(0),ki.stride(1),v.stride(0),v.stride(1),group,splits,pvfp16,threshold,
        num_warps=warps,num_stages=stages)
    _merge[(h*n,)](partial,lse,out,n,splits,num_warps=4)
    return out,counts,executed
