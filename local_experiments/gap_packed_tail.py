"""Experimental algebraically equivalent representative attention.
Predecode rank8 V representatives within each physical page; online softmax over
representatives replaces nested page softmax/Haar accumulation. Floating point
summation changes: requires real quality validation, not deployed automatically.
"""
import math,torch,triton as tr,triton.language as tl
from .compensation_stable_kernels import _round_tf32
from .high_sparse_runtime import _budget_mask
from .compensation_kernels import _merge
@tr.jit
def _decode(BASE,DETAIL,OUT,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);d=i%8;r=i//8%4;p=i//32
    a=tl.load(BASE+p*8+d,i<N,0).to(tl.float32)
    x=tl.load(DETAIL+p*24+d,i<N,0).to(tl.float32)
    y=tl.load(DETAIL+p*24+8+d,i<N,0).to(tl.float32)
    z=tl.load(DETAIL+p*24+16+d,i<N,0).to(tl.float32)
    value=((a+tl.where(r<2,x,-x))+tl.where(r%2==0,y,-y))+tl.where((r==0)|(r==3),z,-z)
    tl.store(OUT+i,value,i<N)
@tr.jit
def _packed_tail(Q,PK,PV,MASK,OUT,LSE,SCALE,KH,KL,VH,VL,PACK_K:tl.constexpr,PACK_V:tl.constexpr,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=64,BR:tl.constexpr=64):
    tile=tl.program_id(0);h=tl.program_id(1);qi=tile*BQ+tl.arange(0,BQ);d=tl.arange(0,D);r=tl.arange(0,BR);v=tl.arange(0,16)
    q=tl.load(Q+qi[:,None]*H*D+h*D+d[None,:],qi[:,None]<N,0).to(tl.float32)
    mi=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,16),tl.float32)
    for first in range(tr.cdiv(P*4,BR)):
        rep=first*BR+r;page=rep//4
        word=tl.load(MASK+(h*(N//64)+tile*BQ//64)*16+page//32)
        omitted=((word>>(page%32))&1)==0
        k=tl.load(PK+(h*P*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
        if PACK_K:
            kh=tl.load(KH+(h*P*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
            kl=tl.load(KL+(h*P*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
        else:
            kh=_round_tf32(k);kl=k-kh
        logits=tl.dot(q,kl,input_precision='tf32');logits=tl.dot(q,kh,logits,input_precision='tf32')*(1.4426950408889634/math.sqrt(D))
        logits=tl.where((rep[None,:]<P*4)&omitted[None,:],logits,-float('inf'))
        new=tl.maximum(mi,tl.max(logits,1));fade=tl.exp2(mi-new);prob=tl.exp2(logits-new[:,None])
        values=tl.load(PV+(h*P*4+rep[:,None])*8+v[None,:],(rep[:,None]<P*4)&(v[None,:]<8),0)
        ph=_round_tf32(prob);pl=prob-ph
        if PACK_V:
            vh=tl.load(VH+(h*P*4+rep[:,None])*8+v[None,:],(rep[:,None]<P*4)&(v[None,:]<8),0)
            vl=tl.load(VL+(h*P*4+rep[:,None])*8+v[None,:],(rep[:,None]<P*4)&(v[None,:]<8),0)
        else:
            vh=_round_tf32(values);vl=values-vh
        update=tl.dot(pl,vl,input_precision='tf32');update=tl.dot(pl,vh,update,input_precision='tf32')
        update=tl.dot(ph,vl,update,input_precision='tf32');update=tl.dot(ph,vh,update,input_precision='tf32')
        acc=acc*fade[:,None]+update;den=den*fade+tl.sum(prob,1);mi=new
    tl.store(OUT+h*N*8+qi[:,None]*8+v[None,:],acc/den[:,None],(qi[:,None]<N)&(v[None,:]<8))
    tl.store(LSE+h*N+qi,mi+tl.log2(den)+4.+tl.load(SCALE+h)*1.4426950408889634,qi<N)

@tr.jit
def _pack(X,HI,LO,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,0);hi=_round_tf32(x)
    tl.store(HI+i,hi,i<N);tl.store(LO+i,x-hi,i<N)
def pack(x,hi,lo):
    _pack[(tr.cdiv(x.numel(),256),)](x,hi,lo,x.numel(),256,num_warps=4,enable_fp_fusion=False)
