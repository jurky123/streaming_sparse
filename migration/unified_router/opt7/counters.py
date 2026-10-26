"""Count selected token costs without writing a float [head,query,page] tensor."""
import torch
import triton as tr
import triton.language as tl
@tr.jit
def _partial(M,C,PART,N:tl.constexpr,B:tl.constexpr,HISTORY_PAGES:tl.constexpr,T:tl.constexpr):
    block=tl.program_id(0);idx=block*T+tl.arange(0,T);page=idx%B
    selected=tl.load(M+idx,idx<N,False);cost=tl.load(C+page,idx<N,0.)
    weight=tl.where(selected,cost,0.)
    tl.store(PART+block*2,tl.sum(tl.where(page<HISTORY_PAGES,weight,0.),0))
    tl.store(PART+block*2+1,tl.sum(tl.where(page>=HISTORY_PAGES,weight,0.),0))
@tr.jit
def _finish(P,D,S:tl.constexpr,PAD:tl.constexpr,NQ:tl.constexpr,KH:tl.constexpr,KC:tl.constexpr):
    i=tl.arange(0,PAD);rh=tl.sum(tl.load(P+i*2,i<S,0.),0);rc=tl.sum(tl.load(P+i*2+1,i<S,0.),0)
    tl.store(D,rh+rc);tl.store(D+1,NQ);tl.store(D+2,rh);tl.store(D+3,rc)
    tl.store(D+4,NQ*(KH+KC));tl.store(D+5,1.)
    tl.store(D+6,(rh+rc)/(NQ*(KH+KC)));tl.store(D+7,rh/(NQ*KH));tl.store(D+8,rc/(NQ*KC));tl.store(D+9,0.)
def counters(mask,costs,kh,kc):
    h,n,b=mask.shape;size=mask.numel();splits=tr.cdiv(size,4096)
    partial=torch.empty((splits,2),device=mask.device,dtype=torch.float32)
    diagnostic=torch.empty((10,),device=mask.device,dtype=torch.float32)
    _partial[(splits,)](mask,costs,partial,size,b,b-74,4096,num_warps=4)
    _finish[(1,)](partial,diagnostic,splits,tr.next_power_of_2(splits),h*n,kh,kc,num_warps=4)
    return diagnostic
