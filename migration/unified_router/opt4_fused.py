"""Remove selected-weight materialization; apply the page mask inside GEMM."""
import torch
import triton as tr
import triton.language as tl
import math

@tr.jit
def _fused_first(L,LC,G,START,COUNT,C,S,W,E,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,Q:tl.constexpr,PREC:tl.constexpr):
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
    tl.store(W+(h*R+start+p[:,None])*N+q[None,:],tl.trans(w)*cost,
             (q[None,:]<N)&(p[:,None]<count))


@tr.jit
def _masked_mean(W,V,MASK,PAGE,OUT,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr,
                 BM:tl.constexpr,BK:tl.constexpr,PREC:tl.constexpr):
    h=tl.program_id(1);m=tl.program_id(0)*BM+tl.arange(0,BM);q=m%N
    k=tl.arange(0,BK);d=tl.arange(0,32)
    acc=tl.full((BM,32),0.,tl.float32)
    for off in range(tr.cdiv(R,BK)):
        r=off*BK+k
        page=tl.load(PAGE+r,r<R,0)
        selected=tl.load(MASK+(h*N+q[:,None])*B+page[None,:],
            (m[:,None]>=N)&(m[:,None]<2*N)&(r[None,:]<R),True)
        a=tl.load(W+(h*R+r[None,:])*N+q[:,None],(m[:,None]<2*N)&(r[None,:]<R),0.)
        a=tl.where(selected,a,0.)
        v=tl.load(V+(h*R+r[:,None])*32+d[None,:],r[:,None]<R,0.)
        if PREC=='fp16':acc=tl.dot(a.to(tl.float16),v.to(tl.float16),acc)
        else:acc=tl.dot(a,v,acc,input_precision=PREC)
    tl.store(OUT+(h*2*N+m[:,None])*32+d[None,:],acc,m[:,None]<2*N)


def first(logits,state,costs,b,precision,tile=64):
    h,n,r=logits.shape
    score=torch.empty((h,n,b),device=logits.device,dtype=torch.float32)
    energy=torch.empty_like(score)
    weights=torch.empty((h,r,n),device=logits.device,dtype=torch.float16 if precision=='fp16'else torch.float32)
    _fused_first[(tr.cdiv(n,tile),b,h)](logits,state['lc'],state['grams'],state['starts'],state['counts'],costs,
        score,weights,energy,n,b,r,tile,precision,num_warps=4,enable_fp_fusion=True)
    return score,weights,energy


def means(weights,mask,state,precision):
    h,n,b=mask.shape;r=weights.shape[1]
    out=torch.empty((h,2*n,32),device=weights.device,dtype=torch.float32)
    _masked_mean[(tr.cdiv(2*n,64),h)](weights,state['value'],mask,state['page_index'],out,n,b,r,64,32,precision,
        num_warps=4,num_stages=2)
    return out


def install():
    from .opt4 import route
    route.stream_first=first;route.stream_means=means


def load_fused(directory,training_seed=0,**options):
    install()
    from .opt4.network import load_compact
    return load_compact(directory,training_seed,**options)


def main():
    from . import fast_bench
    fast_bench.load_fast=load_fused;fast_bench.main()


if __name__=='__main__':main()

