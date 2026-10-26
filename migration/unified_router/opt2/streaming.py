"""Streaming content moments and candidate vectors; no full page-content tensor."""
import math
import torch
import triton as tr
import triton.language as tl
from .score import _content
from .features import _statistics

@tr.jit
def _moments(L,LC,V,M,C,R,A,PART,CAND,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,TOTAL:tl.constexpr,OFFSET:tl.constexpr,
             GROUPS:tl.constexpr,GOFF:tl.constexpr,GP:tl.constexpr,Q:tl.constexpr,PP:tl.constexpr):
    qi=tl.program_id(0)*Q+tl.arange(0,Q);pg=tl.program_id(1);head=tl.program_id(2)
    pr=tl.arange(0,PP);d=tl.arange(0,32);slots=tl.arange(0,8)
    row=head*N+qi
    remove=tl.load(R+row[:,None]*8+slots[None,:],qi[:,None]<N,-1)
    add=tl.load(A+row[:,None]*8+slots[None,:],qi[:,None]<N,-1)
    allnum=tl.full((Q,32),0.,tl.float64);selnum=tl.full((Q,32),0.,tl.float64)
    alle2=tl.full((Q,),0.,tl.float64);sele2=tl.full((Q,),0.,tl.float64)
    allcost=tl.full((Q,),0.,tl.float64);selcost=tl.full((Q,),0.,tl.float64)
    for j in range(GP):
        page=pg*GP+j;valid=page<B;global_page=OFFSET+page
        ll=tl.load(L+((head*N+qi[:,None])*B+page)*P+pr[None,:],(qi[:,None]<N)&valid&(pr[None,:]<P),0.)
        lc=tl.load(LC+(head*B+page)*P+pr,valid&(pr<P),float('-inf'))
        ll=ll/math.sqrt(128)+lc[None,:]
        maximum=tl.max(ll,1);maximum=tl.where(valid,maximum,0.)
        w=tl.exp(ll-maximum[:,None]);denom=tl.sum(w,1)
        w=w/tl.where(valid,denom,1.)[:,None]
        values=tl.load(V+((head*B+page)*P+pr[:,None])*32+d[None,:],valid&(pr[:,None]<P),0.)
        value=tl.dot(w,values,input_precision='ieee')
        cost=tl.load(C+global_page,valid,0.)
        selected=tl.load(M+row*TOTAL+global_page,(qi<N)&valid,False)
        sc=cost*selected
        vd=value.to(tl.float64);e2=tl.sum(vd*vd,1)
        allnum+=vd*cost;selnum+=vd*sc[:,None]
        alle2+=e2*cost;sele2+=e2*sc
        allcost+=cost;selcost+=sc
        ri=tl.min(tl.where(remove==global_page,slots[None,:],8),1)
        aj=tl.min(tl.where(add==global_page,slots[None,:],8),1)
        tl.store(CAND+(row[:,None]*16+ri[:,None])*32+d[None,:],value,(qi[:,None]<N)&valid&(ri[:,None]<8))
        tl.store(CAND+(row[:,None]*16+8+aj[:,None])*32+d[None,:],value,(qi[:,None]<N)&valid&(aj[:,None]<8))
    base=(row[:,None]*GROUPS+GOFF+pg)*68
    tl.store(PART+base+d[None,:],allnum,qi[:,None]<N)
    tl.store(PART+base+32+d[None,:],selnum,qi[:,None]<N)
    tl.store(PART+(row*GROUPS+GOFF+pg)*68+64,alle2,qi<N)
    tl.store(PART+(row*GROUPS+GOFF+pg)*68+65,sele2,qi<N)
    tl.store(PART+(row*GROUPS+GOFF+pg)*68+66,allcost,qi<N)
    tl.store(PART+(row*GROUPS+GOFF+pg)*68+67,selcost,qi<N)

@tr.jit
def _reduce_moments(PART,X,M,C,O,ST,N:tl.constexpr,B:tl.constexpr,BP:tl.constexpr,G:tl.constexpr,GG:tl.constexpr):
    row=tl.program_id(0);g=tl.arange(0,GG);d=tl.arange(0,32);p=tl.arange(0,BP)
    totalnum=tl.sum(tl.load(PART+(row*G+g[:,None])*68+d[None,:],g[:,None]<G,0.),0)
    selnum=tl.sum(tl.load(PART+(row*G+g[:,None])*68+32+d[None,:],g[:,None]<G,0.),0)
    e2=tl.sum(tl.load(PART+(row*G+g)*68+64,g<G,0.),0)
    se2=tl.sum(tl.load(PART+(row*G+g)*68+65,g<G,0.),0)
    cost=tl.sum(tl.load(PART+(row*G+g)*68+66,g<G,0.),0)
    scost=tl.sum(tl.load(PART+(row*G+g)*68+67,g<G,0.),0)
    total=totalnum/cost;center=selnum/scost
    # Float64 moments prevent cancellation from corrupting small set variances.
    da=tl.maximum(e2/cost-tl.sum(total*total,0),0.)
    ds=tl.maximum(se2/scost-tl.sum(center*center,0),0.)
    x=tl.load(X+row*B+p,p<B,0.);s=tl.load(M+row*B+p,p<B,False)
    c=tl.load(C+p,p<B,0.);origin=tl.load(O+p,p<B,0.)
    mean=tl.sum(x,0)/B;std=tl.maximum(tl.sqrt(tl.sum(tl.where(p<B,(x-mean)*(x-mean),0.),0)/B),1.e-6)
    ns=tl.sum(s.to(tl.int32),0);meanz=tl.sum(tl.where(s,(x-mean)/std,0.),0)/ns
    tl.store(ST+row*80+d,total);tl.store(ST+row*80+32+d,center)
    tl.store(ST+row*80+64,ds);tl.store(ST+row*80+65,da)
    tl.store(ST+row*80+66,mean);tl.store(ST+row*80+67,std)
    tl.store(ST+row*80+68,scost);tl.store(ST+row*80+69,cost)
    tl.store(ST+row*80+70,tl.sum(c*s*origin,0));tl.store(ST+row*80+71,ns);tl.store(ST+row*80+72,meanz)

def score_only(q,basis,projected):
    h,n,_=q.shape;total=sum(lc.shape[1]for k,lc,v in projected)
    score=torch.empty((h,n,total),device=q.device,dtype=torch.float32)
    qq=torch.bmm(q.float(),basis);offset=0;raw=[]
    for key,lc,value in projected:
        b,p=lc.shape[1:];logits=torch.bmm(qq,key.transpose(1,2));raw.append(logits)
        _content[(tr.cdiv(n,32),b,h)](logits,lc,None,score,None,n,b,p,total,offset,32,max(16,p),False,
            num_warps=4,enable_fp_fusion=False)
        offset+=b
    return score,raw

def streaming_stats(q,sketch,score,mask,rem,add,raw,projected,gp=16,tile=16):
    h,n,b=score.shape;groups=sum(tr.cdiv(lc.shape[1],gp)for _,lc,_ in projected)
    partial=torch.empty((h*n,groups,68),device=q.device,dtype=torch.float64)
    candidates=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
    stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
    offset=0;goff=0
    for logits,(_,lc,value)in zip(raw,projected):
        nb,p=lc.shape[1:]
        _moments[(tr.cdiv(n,tile),tr.cdiv(nb,gp),h)](logits,lc,value,mask,sketch.costs,rem,add,partial,candidates,
            n,nb,p,b,offset,groups,goff,gp,tile,max(16,p),num_warps=4,num_stages=1,enable_fp_fusion=False)
        offset+=nb;goff+=tr.cdiv(nb,gp)
    _reduce_moments[(h*n,)](partial,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),groups,
        tr.next_power_of_2(groups),num_warps=4,enable_fp_fusion=False)
    return stats,candidates
