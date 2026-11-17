"""Split-K page-LUT exact + R4 tail with complete FP32 representative Values.

Representative Values are page-only reconstructions of full mean + rank8 detail;
no Query×page×128 tensor is formed. Tail P×V defaults to the user-accepted four-BF16-product approximation;
mode1 retains IEEE FP32 as a control. Raw exact uses two BF16 products. The 16-token representative mass is included before normalization.
"""
import torch
import triton as tr
import triton.language as tl
from ..review_v1.ops import haar

@tr.jit
def update(qk,v,m,l,o,FULL:tl.constexpr,TAIL_MODE:tl.constexpr=4):
    new_m=tl.maximum(m,tl.max(qk,1));p=tl.exp2(qk-new_m[:,None]);alpha=tl.exp2(m-new_m)
    o=o*alpha[:,None]
    if FULL:
        if TAIL_MODE==1:
            o+=tl.dot(p,v,input_precision='ieee')
        elif TAIL_MODE:
            ph=p.to(tl.bfloat16);pl=(p-ph.to(tl.float32)).to(tl.bfloat16)
            vh=v.to(tl.bfloat16);vl=(v-vh.to(tl.float32)).to(tl.bfloat16)
            o+=tl.dot(ph,vh)+tl.dot(ph,vl)+tl.dot(pl,vh)
            if TAIL_MODE==4:o+=tl.dot(pl,vl)
        else:o+=tl.dot(p,v,input_precision='tf32x3')
    else:
        hi=p.to(v.dtype);lo=(p-hi.to(tl.float32)).to(v.dtype)
        o+=tl.dot(hi,v)+tl.dot(lo,v)
    l=l*alpha+tl.sum(p,1)
    return new_m,l,o

@tr.jit
def mixed_kernel(Q,K,V,SK,SV,IDS,MASK,PART,PLSE,
                 H:tl.constexpr,N:tl.constexpr,P:tl.constexpr,C:tl.constexpr,S:tl.constexpr,SK0:tl.constexpr,SV0:tl.constexpr,TAIL_MODE:tl.constexpr=4,D:tl.constexpr=128):
    tile=tl.program_id(0);head=tl.program_id(1);split=tl.program_id(2);T:tl.constexpr=N//64
    mrow=tile*64+tl.arange(0,64);col=tl.arange(0,64);d=tl.arange(0,D)
    q=tl.load(Q+(mrow[:,None]*H+head)*D+d[None,:])
    m=tl.full((64,),-float('inf'),tl.float32);l=tl.full((64,),0,tl.float32);o=tl.full((64,D),0,tl.float32)
    for i in range(tr.cdiv(C,S)):
        ci=i*S+split
        page=tl.load(IDS+(head*T+tile)*C+ci,ci<C,0)
        tok=page*64+col
        k=tl.load(K+(tok[None,:]*H+head)*D+d[:,None])
        v=tl.load(V+(tok[:,None]*H+head)*D+d[None,:])
        qk=tl.dot(q,k)*(1.4426950408889634/11.313708498984761)
        qk=tl.where(ci<C,qk,-float('inf'))
        m,l,o=update(qk,v,m,l,o,False)
    for i in range(tr.cdiv(P*4,64*S)):
        token=(split*tr.cdiv(P*4,64*S)+i)*64+col;page=token//4
        chosen=tl.load(MASK+(head*T+tile)*P+page,page<P,1)
        k=tl.load(SK+head*SK0+token[None,:]*D+d[:,None],token[None,:]<P*4,0)
        v=tl.load(SV+head*SV0+token[:,None]*D+d[None,:],token[:,None]<P*4,0)
        qk=tl.dot(q,k)*(1.4426950408889634/11.313708498984761)+4.
        qk=tl.where((token[None,:]<P*4)&(~chosen[None,:]),qk,-float('inf'))
        m,l,o=update(qk,v,m,l,o,True,TAIL_MODE)
    row=((head*T+tile)*S+split)*64+tl.arange(0,64)
    tl.store(PART+row[:,None]*D+d[None,:],o/l[:,None])
    tl.store(PLSE+row,m+tl.log2(l))

@tr.jit
def merge_kernel(PART,PLSE,OUT,H:tl.constexpr,N:tl.constexpr,S:tl.constexpr,D:tl.constexpr=128):
    qb=tl.program_id(0);head=tl.program_id(1);q=qb*4+tl.arange(0,4);s=tl.arange(0,S);d=tl.arange(0,D)
    row=((head*(N//64)+q[None,:]//64)*S+s[:,None])*64+q[None,:]%64
    ls=tl.load(PLSE+row);maximum=tl.max(ls,0);w=tl.exp2(ls-maximum[None,:]);w=w/tl.sum(w,0)[None,:]
    value=tl.load(PART+row[:,:,None]*D+d[None,None,:])
    out=tl.sum(w[:,:,None]*value,0)
    tl.store(OUT+(q[:,None]*H+head)*D+d[None,:],out)

@torch.no_grad()
def full_values(summary,basis,haar_basis=None):
    details=torch.bmm(summary.detail.float().flatten(1,2),basis.bfloat16().float().transpose(1,2)).reshape(*summary.detail.shape[:3],128)
    values=summary.mean.float()[:,:,None]+torch.einsum('ar,hpad->hprd',haar(summary.mean) if haar_basis is None else haar_basis,details)
    return values.contiguous()

KERNEL_STATS={}

@torch.no_grad()
def mixed_attention(q,k,v,summary,values,ids,mask,workspace,splits=4,tail_mode=4,warps=4):
    _,n,h,d=q.shape;p=summary.mean.shape[1];count=ids.shape[-1]
    part=workspace.tensor('mixed_part',(h,n//64,splits,64,d),q.device,torch.float32)
    lse=workspace.tensor('mixed_lse',(h,n//64,splits,64),q.device,torch.float32)
    out=workspace.tensor('mixed_out',q.shape,q.device,q.dtype)
    # Summary slabs may have a larger head stride than the active page prefix.
    sk=summary.keys;sv=values
    compiled=mixed_kernel[(n//64,h,splits)](q,k,v,sk,sv,ids.contiguous(),mask.contiguous(),part,lse,h,n,p,count,splits,sk.stride(0),sv.stride(0),tail_mode,num_warps=warps,num_stages=2)
    KERNEL_STATS[(n,p,splits,tail_mode,warps)]=dict(registers=compiled.n_regs,spills=compiled.n_spills)
    merge_kernel[(n//4,h)](part,lse,out,h,n,splits,num_warps=4)
    return out
