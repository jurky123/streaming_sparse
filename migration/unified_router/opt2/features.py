"""Split set reductions from the action head to reduce register lifetimes."""
import torch
import triton as tr
import triton.language as tl
from .actions import _actions

@tr.jit
def _statistics(X,V,S,C,O,ST,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr):
    row=tl.program_id(0);p=tl.arange(0,P);d=tl.arange(0,32)
    x=tl.load(X+row*B+p,p<B,0.)
    selected=tl.load(S+row*B+p,p<B,False)
    c=tl.load(C+p,p<B,0.);origin=tl.load(O+p,p<B,0.)
    value=tl.load(V+(row*B+p[:,None])*32+d[None,:],p[:,None]<B,0.)
    cost=tl.sum(c,0);sc=c*selected;scost=tl.sum(sc,0)
    total=tl.sum(value*c[:,None],0)/cost
    center=tl.sum(value*sc[:,None],0)/scost
    ds=tl.sum(tl.sum((value-center[None,:])*(value-center[None,:]),1)*sc,0)/scost
    da=tl.sum(tl.sum((value-total[None,:])*(value-total[None,:]),1)*c,0)/cost
    mean=tl.sum(x,0)/B
    std=tl.sqrt(tl.sum(tl.where(p<B,(x-mean)*(x-mean),0.),0)/B)
    std=tl.maximum(std,1.e-6);z=(x-mean)/std
    nselected=tl.sum(selected.to(tl.int32),0)
    meanz=tl.sum(tl.where(selected,z,0.),0)/nselected
    tl.store(ST+row*80+d,total);tl.store(ST+row*80+32+d,center)
    tl.store(ST+row*80+64,ds);tl.store(ST+row*80+65,da)
    tl.store(ST+row*80+66,mean);tl.store(ST+row*80+67,std)
    tl.store(ST+row*80+68,scost);tl.store(ST+row*80+69,cost)
    tl.store(ST+row*80+70,tl.sum(sc*origin,0))
    tl.store(ST+row*80+71,nselected);tl.store(ST+row*80+72,meanz)

def split_finish(q,sketch,score,mask,rem,add,beta,step,value,head,stat_warps=4,action_warps=4,factor=False):
    h,n,b=score.shape
    stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
    out=torch.empty_like(mask)
    _statistics[(h*n,)](score,value,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),
        num_warps=stat_warps,enable_fp_fusion=False)
    from .factor_actions import _factor_actions
    action=_factor_actions if factor else _actions
    action[(h*n,)](stats,score,value,mask,sketch.costs,sketch.origins,rem,add,
        head.w1,head.b1,head.w2,head.b2,head.mean,head.std,step,out,n,b,tr.next_power_of_2(b),
        sketch.history_tokens,sketch.current_tokens,beta,num_warps=action_warps,enable_fp_fusion=False)
    return out
