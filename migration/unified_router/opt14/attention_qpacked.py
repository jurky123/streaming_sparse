"""Read native history/current KV directly, avoiding two BF16 concatenations.
K is quantized directly; optional fused V packing avoids strided PV loads.
"""
import torch
import triton as tr
import triton.language as tl
from ..fast_kernels import _union
from triton.language.extra.cuda.libdevice import nearbyint
@tr.jit
def _split(QP,QS,KP,VHIST,VCUR,KS,M,IDS,COUNTS,OUT,LSE,
               Q:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
               QH:tl.constexpr,QQ:tl.constexpr,KH:tl.constexpr,KK:tl.constexpr,
               VH:tl.constexpr,VK:tl.constexpr,CH:tl.constexpr,CK:tl.constexpr,NH:tl.constexpr,G:tl.constexpr,SPLITS:tl.constexpr,PVFP16:tl.constexpr, PACKV:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2)
    qi=g*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,64)
    q=tl.load(QP+h*QH+qi[:,None]*QQ+d[None,:],qi[:,None]<Q,0)
    if G==32:qs=tl.load(QS+h*tr.cdiv(Q,32)+g)
    else:qs=tl.load(QS+h*tr.cdiv(Q,32)+qi//32,qi<Q,0)[:,None]
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
        ks=tl.load(KS+h*B+page)
        logits=tl.dot(q,k).to(tl.float32)*(qs*ks)*0.12751743082459617
        visible=tl.load(M+(h*Q+qi)*B+page,qi<Q,False)
        logits=tl.where(visible[:,None]&valid[None,:],logits,float('-inf'))
        newmax=tl.maximum(maximum,tl.max(logits,1))
        safe=tl.where(newmax==float('-inf'),0.,newmax)
        alpha=tl.exp2(maximum-safe)
        prob=tl.exp2(logits-safe[:,None])
        if PACKV:
            v=tl.load(VHIST+h*VH+ki[:,None]*VK+d[None,:],valid[:,None],0)
        else:
            vptr=tl.where(ki[:,None]<NH,VHIST+h*VH+ki[:,None]*VK+d[None,:],VCUR+h*CH+(ki[:,None]-NH)*CK+d[None,:])
            v=tl.load(vptr,valid[:,None],0)
        if PVFP16:v=v.to(tl.float16)
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

@tr.jit
def _quantize(KHIST,KCUR,O,S,VHIST,VCUR,VPACK,N:tl.constexpr,B:tl.constexpr,HS:tl.constexpr,NS:tl.constexpr,CH:tl.constexpr,CS:tl.constexpr,NH:tl.constexpr,VHS:tl.constexpr,VNS:tl.constexpr,VCH:tl.constexpr,VCS:tl.constexpr,FUSE:tl.constexpr):
    h=tl.program_id(1);page=tl.program_id(0);t=tl.arange(0,64);d=tl.arange(0,128)
    ki=page//74*4680+page%74*64+t;valid=(page%74*64+t<4680)&(ki<N)
    ptr=tl.where(ki[:,None]<NH,KHIST+h*HS+ki[:,None]*NS+d[None,:],KCUR+h*CH+(ki[:,None]-NH)*CS+d[None,:])
    k=tl.load(ptr,valid[:,None],0.).to(tl.float32)
    scale=tl.maximum(tl.max(tl.max(tl.abs(k),1),0)/127.,1.e-10)
    quant=nearbyint(k/scale).to(tl.int8)
    tl.store(O+(h*N+ki[:,None])*128+d[None,:],quant,valid[:,None])
    tl.store(S+h*B+page,scale)
    if FUSE:
        vp=tl.where(ki[:,None]<NH,VHIST+h*VHS+ki[:,None]*VNS+d[None,:],VCUR+h*VCH+(ki[:,None]-NH)*VCS+d[None,:])
        vv=tl.load(vp,valid[:,None],0)
        tl.store(VPACK+(h*N+ki[:,None])*128+d[None,:],vv,valid[:,None])

@tr.jit
def _quantize_q(Q,O,S,N:tl.constexpr,HS:tl.constexpr,NS:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);qi=g*32+tl.arange(0,32);d=tl.arange(0,128)
 q=tl.load(Q+h*HS+qi[:,None]*NS+d[None,:],qi[:,None]<N,0).to(tl.float32)
 scale=tl.maximum(tl.max(tl.max(tl.abs(q),1),0)/127.,1.e-10)
 quant=nearbyint(q/scale).to(tl.int8)
 tl.store(O+(h*N+qi[:,None])*128+d[None,:],quant,qi[:,None]<N)
 tl.store(S+h*tr.cdiv(N,32)+g,scale)

def attention(q,kh,vh,kc,vc,mask,group=32,splits=2,warps=2,stages=2,return_union=False,pack_v=False,fuse_pack=False):
    h,n,d=q.shape;b=mask.shape[-1];nh=kh.shape[1];nk=nh+kc.shape[1]
    qp=torch.empty((h,n,d),device=q.device,dtype=torch.int8);qs=torch.empty((h,tr.cdiv(n,32)),device=q.device,dtype=torch.float32)
    _quantize_q[(tr.cdiv(n,32),h)](q,qp,qs,n,q.stride(0),q.stride(1),num_warps=4)
    original_vh=vh
    if pack_v:
        vh=torch.empty((h,nk,d),device=q.device,dtype=vh.dtype)if fuse_pack else torch.cat((vh,vc),1)
    ki=torch.empty((h,nk,d),device=q.device,dtype=torch.int8)
    ks=torch.empty((h,b),device=q.device,dtype=torch.float32)
    _quantize[(b,h)](kh,kc,ki,ks,original_vh,vc,vh,nk,b,kh.stride(0),kh.stride(1),kc.stride(0),kc.stride(1),nh,original_vh.stride(0),original_vh.stride(1),vc.stride(0),vc.stride(1),pack_v and fuse_pack,num_warps=4)
    ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32)
    counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
    _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
    lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
    _split[(tr.cdiv(n,group),h,splits)](qp,qs,ki,vh,vc,ks,mask,ids,counts,partial,lse,n,nk,b,
        qp.stride(0),qp.stride(1),ki.stride(0),ki.stride(1),vh.stride(0),vh.stride(1),vc.stride(0),vc.stride(1),nh,group,splits,False,pack_v,
        num_warps=warps,num_stages=stages)
    _merge[(h*n,)](partial,lse,out,n,splits,num_warps=4)
    return (out,counts)if return_union else out
