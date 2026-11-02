"""Query-dependent page-local V: 3 Haar details or one selected detail; same Z."""
import math
import torch,triton as tr,triton.language as tl
from .compensation_stable_kernels import _round_tf32
from .compensation_ordered_kernels import _sum4
from .high_sparse_runtime import _budget_mask
from .compensation_kernels import _merge

@tr.jit
def _reload_tail(Q,PK,BASE,DETAIL,CHOICE,MASK,OUT,LSE,SCALE,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,MODE:tl.constexpr,BQ:tl.constexpr=64,BP:tl.constexpr=16):
    tile=tl.program_id(0);h=tl.program_id(1);qi=tile*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,16);pages=tl.arange(0,BP);rr=tl.arange(0,BP*4)
    mi=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,16),tl.float32)
    for first in range(tr.cdiv(P,BP)):
        ids=first*BP+pages;rep=first*BP*4+rr
        word=tl.load(MASK+(h*(N//64)+tile*BQ//64)*16+ids//32);omitted=((word>>(ids%32))&1)==0
        q=tl.load(Q+qi[:,None]*H*D+h*D+d[None,:],qi[:,None]<N,0,volatile=True).to(tl.float32)
        keys=tl.load(PK+(h*P*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
        hi=_round_tf32(keys);lo=keys-hi;logits=tl.dot(q,lo,input_precision='tf32');logits=tl.dot(q,hi,logits,input_precision='tf32')*(1.4426950408889634/math.sqrt(D))
        logits=tl.reshape(logits,(BQ,BP,4));mx=tl.max(logits,2);e=tl.exp2(logits-mx[:,:,None]);es=_sum4(e,False,BQ,BP)
        z=mx+tl.log2(es)+4.;z=tl.where((ids[None,:]<P)&omitted[None,:],z,-float('inf'));new=tl.maximum(mi,tl.max(z,1));fade=tl.exp2(mi-new);prob=tl.exp2(z-new[:,None])
        base=tl.load(BASE+(h*P+ids[:,None])*8+r[None,:],(ids[:,None]<P)&(r[None,:]<8),0).to(tl.float32)
        hp=_round_tf32(prob);lp=prob-hp;update=tl.dot(lp,base,input_precision='tf32');update=tl.dot(hp,base,update,input_precision='tf32')
        if MODE==1:
            choice=tl.load(CHOICE+h*P+ids,ids<P,0).to(tl.int32)
            rid=tl.arange(0,4)
            signs=tl.where(choice[:,None]==0,tl.where(rid[None,:]<2,1.,-1.),tl.where(choice[:,None]==1,tl.where(rid[None,:]%2==0,1.,-1.),tl.where((rid[None,:]==0)|(rid[None,:]==3),1.,-1.)))
            contrast=_sum4(e*signs[None,:,:],False,BQ,BP)/es;w=prob*contrast
            value=tl.load(DETAIL+(h*P+ids[:,None])*8+r[None,:],(ids[:,None]<P)&(r[None,:]<8),0).to(tl.float32)
            hw=_round_tf32(w);lw=w-hw;part=tl.dot(lw,value,input_precision='tf32');part=tl.dot(hw,value,part,input_precision='tf32');update+=part
        else:
            for mode in tl.static_range(3):
                rid=tl.arange(0,4)
                if mode==0:signs=tl.where(rid<2,1.,-1.)
                elif mode==1:signs=tl.where(rid%2==0,1.,-1.)
                else:signs=tl.where((rid==0)|(rid==3),1.,-1.)
                contrast=_sum4(e*signs[None,None,:],False,BQ,BP)/es;w=prob*contrast
                value=tl.load(DETAIL+((h*P+ids[:,None])*3+mode)*8+r[None,:],(ids[:,None]<P)&(r[None,:]<8),0).to(tl.float32)
                hw=_round_tf32(w);lw=w-hw;part=tl.dot(lw,value,input_precision='tf32');part=tl.dot(hw,value,part,input_precision='tf32');update+=part
        acc=acc*fade[:,None]+update;den=den*fade+tl.sum(prob,1);mi=new
    tl.store(OUT+h*N*8+qi[:,None]*8+r[None,:],acc/den[:,None],(qi[:,None]<N)&(r[None,:]<8))
    shift=tl.load(SCALE+h)*1.4426950408889634
    tl.store(LSE+h*N+qi,mi+tl.log2(den)+shift,qi<N)
