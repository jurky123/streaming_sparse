"""Compute score, prototype weights and energy once; apply selection later.

Weights are query-contiguous [head, prototype, 2*query], enabling coalesced
writes from the page-local softmax and a transposed input to the mean GEMM.
"""
import math
import torch
import triton as tr
import triton.language as tl
from .packed import prepare_packed
from .kernels import _mm


def prepare_stream(prepared):
    state=prepare_packed(prepared)
    ids=[];offset=0
    for _,lc,_ in prepared[1]:
        b,p=lc.shape[1:]
        ids.append((torch.arange(b,device=lc.device,dtype=torch.int32)+offset).repeat_interleave(p))
        offset+=b
    state['page_index']=torch.cat(ids)
    return state


@tr.jit
def _first(L,LC,G,START,COUNT,C,S,W,E,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,Q:tl.constexpr,PREC:tl.constexpr):
    q=tl.program_id(0)*Q+tl.arange(0,Q);page=tl.program_id(1);h=tl.program_id(2)
    p=tl.arange(0,16);start=tl.load(START+page);count=tl.load(COUNT+page)
    ll=tl.load(L+(h*N+q[:,None])*R+start+p[None,:],(q[:,None]<N)&(p[None,:]<count),0.)
    lc=tl.load(LC+h*R+start+p,p<count,float('-inf'))
    ll=ll/math.sqrt(128)+lc[None,:]
    maximum=tl.max(ll,1);w=tl.exp(ll-maximum[:,None]);den=tl.sum(w,1)
    s=maximum+tl.log(den);w=w/den[:,None]
    gram=tl.load(G+(h*B+page)*256+p[:,None]*16+p[None,:])
    if PREC=='fp16':wg=tl.dot(w.to(tl.float16),gram.to(tl.float16))
    else:wg=tl.dot(w,gram,input_precision=PREC)
    energy=tl.sum(wg*w,1)
    tl.store(S+(h*N+q)*B+page,s,q<N);tl.store(E+(h*N+q)*B+page,energy,q<N)
    cost=tl.load(C+page)
    tl.store(W+(h*R+start+p[:,None])*2*N+q[None,:],tl.trans(w)*cost,
             (q[None,:]<N)&(p[:,None]<count))


@tr.jit
def _select(W,M,PAGE,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,H:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);q=i%N;r=(i//N)%R;h=i//(N*R)
    page=tl.load(PAGE+r,i<H*N*R,0)
    selected=tl.load(M+(h*N+q)*B+page,i<H*N*R,False)
    val=tl.load(W+(h*R+r)*2*N+q,i<H*N*R,0.)
    tl.store(W+(h*R+r)*2*N+N+q,val*selected,i<H*N*R)


def first(logits,state,costs,b,precision,tile=64):
    h,n,r=logits.shape
    score=torch.empty((h,n,b),device=logits.device,dtype=torch.float32)
    energy=torch.empty_like(score)
    weights=torch.empty((h,r,2*n),device=logits.device,dtype=torch.float16 if precision=='fp16'else torch.float32)
    _first[(tr.cdiv(n,tile),b,h)](logits,state['lc'],state['grams'],state['starts'],state['counts'],costs,
        score,weights,energy,n,b,r,tile,precision,num_warps=4,enable_fp_fusion=True)
    return score,weights,energy


def means(weights,mask,state,precision):
    h,n,b=mask.shape;r=weights.shape[1]
    _select[(tr.cdiv(h*n*r,1024),)](weights,mask,state['page_index'],n,b,r,h,1024,num_warps=4)
    a=weights.transpose(1,2);v=state['value']
    out=torch.empty((h,2*n,32),device=weights.device,dtype=torch.float32)
    _mm[(tr.cdiv(2*n,64),1,h)](a,v,out,2*n,32,r,*a.stride(),*v.stride(),64,32,32,precision,
        num_warps=4,num_stages=2)
    return out
