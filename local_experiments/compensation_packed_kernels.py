"""FP32 cached keys decomposed into 3 BF16 operands; quality must be revalidated."""
import math
import torch
import triton as tr
import triton.language as tl
from .compensation_kernels import _merge
from .compensation_fast_kernels import mask

@tr.jit
def _pack(PK,PACKED,P:tl.constexpr,CAP:tl.constexpr,D:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);size=H*CAP*4*D
    active=(i<size)&((i//D)%(CAP*4)<P*4)
    x=tl.load(PK+i,active,0)
    high=x.to(tl.bfloat16);res=x-high.to(tl.float32)
    mid=res.to(tl.bfloat16);low=(res-mid.to(tl.float32)).to(tl.bfloat16)
    tl.store(PACKED+i,high,i<size);tl.store(PACKED+size+i,mid,i<size);tl.store(PACKED+2*size+i,low,i<size)

@tr.jit
def _packed_tail(Q,PK,CODE,MASK,OUT,LSE,P:tl.constexpr,CAP:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr,BP:tl.constexpr,DIRECT:tl.constexpr):
    tile=tl.program_id(0);h=tl.program_id(1)
    qi=tile*BQ+tl.arange(0,BQ);di=tl.arange(0,D)
    q=tl.load(Q+qi[:,None]*(H*D)+h*D+di[None,:],qi[:,None]<N,0).to(tl.float32)
    mi=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,16),tl.float32)
    rr=tl.arange(0,BP*4);rank=tl.arange(0,16);pages=tl.arange(0,BP)
    for first in range(tr.cdiv(P,BP)):
        ids=first*BP+pages;rep=first*BP*4+rr
        word=tl.load(MASK+(h*(N//64)+tile*BQ//64)*16+ids//32)
        omitted=((word>>(ids%32))&1)==0
        size=H*CAP*4*D
        offsets=h*CAP*4*D+rep[None,:]*D+di[:,None]
        k2=tl.load(PK+2*size+offsets,rep[None,:]<P*4,0)
        logits=tl.dot(q.to(tl.bfloat16),k2)
        k1=tl.load(PK+size+offsets,rep[None,:]<P*4,0)
        logits=tl.dot(q.to(tl.bfloat16),k1,logits)
        k0=tl.load(PK+offsets,rep[None,:]<P*4,0)
        logits=tl.dot(q.to(tl.bfloat16),k0,logits)*(1.4426950408889634/math.sqrt(D))
        if DIRECT:
            grouped=tl.reshape(logits,(BQ,BP,4))
            z=tl.reshape(tl.where((ids[None,:,None]<P)&omitted[None,:,None],grouped+4.,-float('inf')),(BQ,BP*4))
            new=tl.maximum(mi,tl.max(z,1));scale=tl.exp2(mi-new)
            prob=tl.sum(tl.reshape(tl.exp2(z-new[:,None]),(BQ,BP,4)),2)
        else:
            logits=tl.reshape(logits,(BQ,BP,4))
            mx=tl.max(logits,2);z=mx+tl.log2(tl.sum(tl.exp2(logits-mx[:,:,None]),2))+4.
            z=tl.where((ids[None,:]<P)&omitted[None,:],z,-float('inf'))
            new=tl.maximum(mi,tl.max(z,1));scale=tl.exp2(mi-new);prob=tl.exp2(z-new[:,None])
        codes=tl.load(CODE+h*CAP*8+ids[:,None]*8+rank[None,:],(ids[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
        acc=acc*scale[:,None]+tl.dot(prob,codes,input_precision='tf32x3')
        den=den*scale+tl.sum(prob,1);mi=new
    tl.store(OUT+h*N*8+qi[:,None]*8+rank[None,:],acc/den[:,None],(qi[:,None]<N)&(rank[None,:]<8))
    tl.store(LSE+h*N+qi,mi+tl.log2(den),qi<N)

def pack(pk,p):
    out=torch.empty((3,)+tuple(pk.shape),device=pk.device,dtype=torch.bfloat16)
    _pack[(tr.cdiv(pk.numel(),1024),)](pk,out,p,pk.shape[1],pk.shape[-1],pk.shape[0],1024,num_warps=4)
    return out

def tail(q,pk,code,bits,p,config):
    _,n,h,d=q.shape;bq,bp,warps,stages,direct=config
    out=torch.empty(h,n,8,device=q.device);lse=torch.empty(h,n,device=q.device)
    compiled=_packed_tail[(tr.cdiv(n,bq),h)](q,pk,code,bits,out,lse,p,pk.shape[2],n,h,d,bq,bp,bool(direct),num_warps=warps,num_stages=stages)
    return out,lse,compiled

def compensate(q,pk,code,lut,exact,lse,basis,center,gamma,p,config=(16,16,4,1,False)):
    _,n,h,d=q.shape;bits=mask(lut,n);packed=pack(pk,p);out,lt,_=tail(q,packed,code,bits,p,config)
    _merge[(tr.cdiv(n,16),h)](exact,lse,out,lt,basis,center,gamma,exact,n,h,d,16,num_warps=4)
    return exact,out,lt
