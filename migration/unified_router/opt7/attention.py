"""Multipage streaming attention; preserves every query's independent mask."""
import torch
import triton as tr
import triton.language as tl
from ..fast_kernels import _union

@tr.jit
def _multipage(QP,KP,VP,M,IDS,COUNTS,OUT,Q:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
               QH:tl.constexpr,QQ:tl.constexpr,KH:tl.constexpr,KK:tl.constexpr,VH:tl.constexpr,VK:tl.constexpr,
               G:tl.constexpr,BN:tl.constexpr,PIPE:tl.constexpr,CACHE:tl.constexpr):
    group=tl.program_id(0);h=tl.program_id(1)
    qi=group*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,BN)
    q=tl.load(QP+h*QH+qi[:,None]*QQ+d[None,:],qi[:,None]<Q,0)
    maximum=tl.full((G,),float('-inf'),tl.float32);denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
    count=tl.load(COUNTS+h*tr.cdiv(Q,G)+group)
    for j in tl.range(0,tl.cdiv(count,BN//64),num_stages=PIPE):
        slot=j*(BN//64)+tl.arange(0,BN//64)
        page=tl.load(IDS+(h*tr.cdiv(Q,G)+group)*B+slot,slot<count,0)
        local=page%74
        ki=(page[:,None]//74*4680+local[:,None]*64+tl.arange(0,64)[None,:]).reshape((BN,))
        valid=((slot[:,None]<count)&(local[:,None]*64+tl.arange(0,64)[None,:]<4680)).reshape((BN,))&(ki<K)
        k=tl.load(KP+h*KH+ki[None,:]*KK+d[:,None],valid[None,:],0,cache_modifier=CACHE)
        logits=tl.dot(q,k)*0.12751743082459617
        visible=tl.load(M+(h*Q+qi[:,None])*B+page[None,:],(qi[:,None]<Q)&(slot[None,:]<count),False)
        visible=tl.broadcast_to(visible[:,:,None],(G,BN//64,64)).reshape((G,BN))
        logits=tl.where(visible&valid[None,:],logits,float('-inf'))
        newmax=tl.maximum(maximum,tl.max(logits,1));safe=tl.where(newmax==float('-inf'),0.,newmax)
        alpha=tl.exp2(maximum-safe);prob=tl.exp2(logits-safe[:,None])
        v=tl.load(VP+h*VH+ki[:,None]*VK+d[None,:],valid[:,None],0,cache_modifier=CACHE)
        acc=tl.dot(prob.to(v.dtype),v,acc*alpha[:,None])
        denom=denom*alpha+tl.sum(prob,1);maximum=newmax
    tl.store(OUT+(h*Q+qi[:,None])*128+d[None,:],acc/denom[:,None],qi[:,None]<Q)

def attention(q,k,v,mask,group=32,bn=128,warps=4,stages=3,pipe=3,cache='',return_union=False):
    h,n,d=q.shape;b=mask.shape[-1]
    ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32)
    counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
    _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    _multipage[(tr.cdiv(n,group),h)](q,k,v,mask,ids,counts,out,n,k.shape[1],b,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),group,bn,pipe,cache,
        num_warps=warps,num_stages=stages)
    return (out,counts)if return_union else out
