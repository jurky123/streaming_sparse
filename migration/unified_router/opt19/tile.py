"""Explicit tile budgets from mean per-query proxy probability or mask votes."""
import math
import torch
import triton as tr
import triton.language as tl
from ..opt8.projection import project
from ..opt17.route import _landmark,_coarse,_top,_fine,_chosen,_select


@tr.jit
def _tiles(S, IDS, OUT, N:tl.constexpr, B:tl.constexpr, C:tl.constexpr,
           NG:tl.constexpr, TG:tl.constexpr, G:tl.constexpr, KEEP:tl.constexpr,
           CC:tl.constexpr, EXCLUDE_TAIL:tl.constexpr):
    tile=tl.program_id(0);h=tl.program_id(1)
    qi=tile*G+tl.arange(0,G);c=tl.arange(0,CC)
    score=tl.load(S+(h*N+qi[:,None])*C+c[None,:],(qi[:,None]<N)&(c[None,:]<C),float('-inf'))
    mx=tl.max(score,1);mx=tl.where(qi<N,mx,0.)
    prob=tl.exp(score-mx[:,None]);prob=prob/tl.maximum(tl.sum(prob,1),1e-20)[:,None]
    importance=tl.sum(prob,0)
    page=tl.load(IDS+(h*NG+(tile*G)//32)*C+c,c<C,0)
    valid=(c<C)
    if EXCLUDE_TAIL:valid=valid&(page%74!=73)
    importance=tl.where(valid,importance,float('-inf'))
    chosen=_chosen(importance,c,C,KEEP)&valid
    rank=tl.cumsum(chosen.to(tl.int32),0)-1
    tl.store(OUT+(h*TG+tile)*KEEP+rank,page,chosen)


@tr.jit
def _expand(IDS,MASK,N:tl.constexpr,B:tl.constexpr,G:tl.constexpr,
            TG:tl.constexpr,K:tl.constexpr,KK:tl.constexpr):
    tile=tl.program_id(0);h=tl.program_id(1);q=tile*G+tl.arange(0,G);j=tl.arange(0,KK)
    page=tl.load(IDS+(h*TG+tile)*K+j,j<K,0)
    tl.store(MASK+(h*N+q[:,None])*B+page[None,:],True,(q[:,None]<N)&(j[None,:]<K))

def expand(ids,n,b,group):
    h,t,k=ids.shape
    mask=torch.zeros((h,n,b),device=ids.device,dtype=torch.bool)
    _expand[(t,h)](ids,mask,n,b,group,t,k,tr.next_power_of_2(k),num_warps=4)
    return mask


def scores(q,state,fraction,reps):
    h,n,_=q.shape;b=state['key'].shape[1];g=tr.cdiv(n,32);c=math.ceil(b*fraction)
    qq=project(q,state['basis'])
    lm=torch.empty((h,g,reps,64),device=q.device,dtype=torch.float16)
    _landmark[(g,h)](qq,lm,n,g,reps,num_warps=4)
    coarse=torch.empty((h,g,b),device=q.device)
    off=0
    for kk,lc in state['parts']:
        np,p=lc.shape[1:];gp=64//p
        _coarse[(g,tr.cdiv(np,gp),h)](lm,kk,lc,coarse,n,b,g,off,np,p,gp,reps,num_warps=4,num_stages=1)
        off+=np
    ids=torch.empty((h,g,c),device=q.device,dtype=torch.int32)
    _top[(h*g,)](coarse,ids,h*g,b,c,tr.next_power_of_2(b),num_warps=4)
    score=torch.empty((h,n,c),device=q.device,dtype=torch.float32)
    _fine[(g,tr.cdiv(c,4),h)](qq,state['key'],state['logcount'],ids,score,n,b,c,g,4,num_warps=4,num_stages=1)
    return score,ids


def select(score,candidates,b,group=32,budget=.15,exclude_tail=False):
    assert group in (16,32)
    h,n,c=score.shape;t=tr.cdiv(n,group);keep=math.ceil(b*budget)
    assert keep<=c-(tr.cdiv(b,74) if exclude_tail else 0)
    ids=torch.empty((h,t,keep),device=score.device,dtype=torch.int32)
    _tiles[(t,h)](score,candidates,ids,n,b,c,candidates.shape[1],t,group,keep,
                  tr.next_power_of_2(c),exclude_tail,num_warps=4)
    return ids


def route(q,state,fraction,reps,group=32,budget=.15,exclude_tail=False,return_scores=False):
    fine,candidates=scores(q,state,fraction,reps)
    ids=select(fine,candidates,state['key'].shape[1],group,budget,exclude_tail)
    return (ids,fine,candidates) if return_scores else ids
