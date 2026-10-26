"""Split the ragged KV loop, then merge partial online-softmax results."""
import torch
import triton as tr
import triton.language as tl
from ..fast_kernels import _union
@tr.jit
def _split(QP,KP,VP,M,IDS,COUNTS,OUT,LSE,
               Q:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
               QH:tl.constexpr,QQ:tl.constexpr,KH:tl.constexpr,KK:tl.constexpr,
               VH:tl.constexpr,VK:tl.constexpr,G:tl.constexpr,SPLITS:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2)
    qi=g*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,64)
    q=tl.load(QP+h*QH+qi[:,None]*QQ+d[None,:],qi[:,None]<Q,0)
    maximum=tl.full((G,),float('-inf'),tl.float32)
    denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
    count=tl.load(COUNTS+h*tr.cdiv(Q,G)+g)
    part=tl.cdiv(count,SPLITS)
    for j in range(split*part,tl.minimum((split+1)*part,count)):
        page=tl.load(IDS+(h*tr.cdiv(Q,G)+g)*B+j)
        # 4680-token segments have 73 full pages and one 8-token tail.
        segment=page//74;local=page%74;ki=segment*4680+local*64+t
        valid=(ki<K)&(local*64+t<4680)
        k=tl.load(KP+h*KH+ki[None,:]*KK+d[:,None],valid[None,:],0)
        logits=tl.dot(q,k).to(tl.float32)*0.12751743082459617 # log2(e)/sqrt(128)
        visible=tl.load(M+(h*Q+qi)*B+page,qi<Q,False)
        logits=tl.where(visible[:,None]&valid[None,:],logits,float('-inf'))
        newmax=tl.maximum(maximum,tl.max(logits,1))
        safe=tl.where(newmax==float('-inf'),0.,newmax)
        alpha=tl.exp2(maximum-safe)
        prob=tl.exp2(logits-safe[:,None])
        v=tl.load(VP+h*VH+ki[:,None]*VK+d[None,:],valid[:,None],0)
        acc=acc*alpha[:,None]+tl.dot(prob.to(v.dtype),v)
        denom=denom*alpha+tl.sum(prob,1);maximum=newmax
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

def attention(q,k,v,mask,group=32,splits=4,warps=4,stages=3,return_union=False):
    h,n,d=q.shape;b=mask.shape[-1]
    ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32)
    counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
    _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
    lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
    _split[(tr.cdiv(n,group),h,splits)](q,k,v,mask,ids,counts,partial,lse,n,k.shape[1],b,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),group,splits,
        num_warps=warps,num_stages=stages)
    _merge[(h*n,)](partial,lse,out,n,splits,num_warps=4)
    return (out,counts)if return_union else out
