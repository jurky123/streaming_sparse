"""Group coarse routing and independent query refinement, without prototype W.

Coarse candidates use four actual queries per packed tile, not shared final masks.
Only candidates receive full P16/P8/P4/P2 log-sum-exp scores. Per-query top-k is
then expanded into the existing attention mask interface. No AV action head yet.
"""
import math
import torch
import triton as tr
import triton.language as tl
from ..opt8.projection import project

@tr.jit
def _chosen(score,index,VALID:tl.constexpr,KEEP:tl.constexpr):
    # Integer order is exact for finite FP32 scores. This Triton version's float
    # sort uses multiplication during swaps: padding +/-inf would produce NaNs.
    bits=score.to(tl.uint32,bitcast=True)
    ordered=tl.where((bits&0x80000000)!=0,~bits,bits^0x80000000)
    sorted_score=tl.sort(ordered,descending=True)
    cutoff=tl.max(tl.where(index==KEEP-1,sorted_score,0),0)
    above=(ordered>cutoff)&(index<VALID)
    equal=(ordered==cutoff)&(index<VALID)
    need=KEEP-tl.sum(above.to(tl.int32),0)
    return above|(equal&(tl.cumsum(equal.to(tl.int32),0)<=need))

@tr.jit
def _landmark(Q,L,N:tl.constexpr,G:tl.constexpr):
    h=tl.program_id(1);g=tl.program_id(0);r=tl.arange(0,16);d=tl.arange(0,64)
    # Four quadrature positions, duplicate safely in the incomplete last tile.
    qi=tl.minimum(g*32+(r%4)*8+3,N-1)
    q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:])
    tl.store(L+((h*G+g)*4+r[:,None])*64+d[None,:],q,r[:,None]<4)

@tr.jit
def _coarse(Q,K,LC,S,N:tl.constexpr,B:tl.constexpr,G:tl.constexpr,OFF:tl.constexpr,NP:tl.constexpr,P:tl.constexpr,GP:tl.constexpr):
    g=tl.program_id(0);pg=tl.program_id(1)*GP+tl.arange(0,GP);h=tl.program_id(2)
    qr=tl.arange(0,16);d=tl.arange(0,64);pp=tl.arange(0,GP*P)
    q=tl.load(Q+((h*G+g)*4+qr[:,None])*64+d[None,:],qr[:,None]<4,0)
    pages=tl.program_id(1)*GP+pp//P
    k=tl.load(K+(h*NP*P+pages[None,:]*P+pp[None,:]%P)*64+d[:,None],pages[None,:]<NP,0)
    logits=tl.dot(q,k).reshape((16,GP,P))*0.08838834764831845
    lc=tl.load(LC+(h*NP+pg[:,None])*P+tl.arange(0,P)[None,:],pg[:,None]<NP,float('-inf'))
    logits=logits+lc[None,:,:]
    mx=tl.max(logits,2);safe=tl.where(mx==float('-inf'),0.,mx)
    score=safe+tl.log(tl.sum(tl.exp(logits-safe[:,:,None]),2))
    score=tl.max(tl.where(qr[:,None]<4,score,float('-inf')),0)
    tl.store(S+(h*G+g)*B+OFF+pg,score,pg<NP)

@tr.jit
def _top(S,I,ROWS:tl.constexpr,B:tl.constexpr,C:tl.constexpr,BB:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,BB)
    score=tl.load(S+row*B+p,p<B,float('-inf'))
    chosen=_chosen(score,p,B,C)
    rank=tl.cumsum(chosen.to(tl.int32),0)-1
    tl.store(I+row*C+rank,p,chosen)

@tr.jit
def _fine(Q,K,LC,IDS,S,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,GP:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(2);ci=tl.program_id(1)*GP+tl.arange(0,GP)
    p=tl.arange(0,GP*16);d=tl.arange(0,64);qi=g*32+tl.arange(0,32)
    page=tl.load(IDS+(h*G+g)*C+tl.program_id(1)*GP+p//16,(tl.program_id(1)*GP+p//16)<C,0)
    q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0)
    k=tl.load(K+((h*B+page[None,:])*16+p[None,:]%16)*64+d[:,None])
    ll=tl.dot(q,k).reshape((32,GP,16))*0.08838834764831845
    pc=tl.load(IDS+(h*G+g)*C+ci,ci<C,0)
    lc=tl.load(LC+(h*B+pc[:,None])*16+tl.arange(0,16)[None,:])
    ll=ll+lc[None,:,:];mx=tl.max(ll,2)
    score=mx+tl.log(tl.sum(tl.exp(ll-mx[:,:,None]),2))
    tl.store(S+(h*N+qi[:,None])*C+ci[None,:],score,(qi[:,None]<N)&(ci[None,:]<C))

@tr.jit
def _select(S,IDS,M,N:tl.constexpr,B:tl.constexpr,C:tl.constexpr,G:tl.constexpr,KEEP:tl.constexpr,CC:tl.constexpr):
    row=tl.program_id(0);h=row//N;q=row%N;c=tl.arange(0,CC)
    score=tl.load(S+row*C+c,c<C,float('-inf'))
    selected=_chosen(score,c,C,KEEP)
    page=tl.load(IDS+(h*G+q//32)*C+c,c<C,0)
    tl.store(M+row*B+page,True,selected)

def prepare(prepared):
    basis,parts=prepared
    keys=[];counts=[];compact=[]
    for k,lc,_ in parts:
        h,b,p=lc.shape
        kk=k.reshape(h,b,p,64).half().contiguous(); ll=lc.contiguous()
        compact.append((kk,ll))
        keys.append(torch.nn.functional.pad(kk,(0,0,0,16-p)))
        counts.append(torch.nn.functional.pad(ll,(0,16-p),value=-torch.inf))
    return dict(basis=basis,parts=compact,key=torch.cat(keys,1).contiguous(),logcount=torch.cat(counts,1).contiguous())

def route(q,state,keep=.15,candidate_fraction=.5,return_aux=False):
    if not 0 < keep <= candidate_fraction <= 1:
        raise ValueError('Require 0 < keep <= candidate_fraction <= 1')
    h,n,_=q.shape;b=state['key'].shape[1];g=tr.cdiv(n,32)
    count=min(b,max(1,math.ceil(b*keep)));c=min(b,max(count,math.ceil(b*candidate_fraction)))
    qq=project(q,state['basis'])
    if c==b:
        ids=torch.arange(b,device=q.device,dtype=torch.int32)[None,None].expand(h,g,b).contiguous()
    else:
        lm=torch.empty((h,g,4,64),device=q.device,dtype=torch.float16)
        _landmark[(g,h)](qq,lm,n,g,num_warps=4)
        coarse=torch.empty((h,g,b),device=q.device)
        off=0
        for kk,lc in state['parts']:
            np,p=lc.shape[1:];gp=64//p
            _coarse[(g,tr.cdiv(np,gp),h)](lm,kk,lc,coarse,n,b,g,off,np,p,gp,num_warps=4,num_stages=1)
            off+=np
        ids=torch.empty((h,g,c),device=q.device,dtype=torch.int32)
        _top[(h*g,)](coarse,ids,h*g,b,c,tr.next_power_of_2(b),num_warps=4)
    scores=torch.empty((h,n,c),device=q.device,dtype=torch.float32)
    _fine[(g,tr.cdiv(c,4),h)](qq,state['key'],state['logcount'],ids,scores,n,b,c,g,4,num_warps=4,num_stages=1)
    mask=torch.zeros((h,n,b),device=q.device,dtype=torch.bool)
    _select[(h*n,)](scores,ids,mask,n,b,c,g,count,tr.next_power_of_2(c),num_warps=4)
    return (mask,dict(candidates=ids,scores=scores,keep_pages=count)) if return_aux else mask
