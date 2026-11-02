"""Rank-8 omitted-page attention with shared online normalization; no teacher."""
import math
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _tail(Q,PK,CODE,LUT,OUT,LSE,P:tl.constexpr,CAP:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr,BP:tl.constexpr):
    tile=tl.program_id(0);h=tl.program_id(1)
    qi=tile*BQ+tl.arange(0,BQ);di=tl.arange(0,D)
    q=tl.load(Q+qi[:,None]*(H*D)+h*D+di[None,:],qi[:,None]<N,0).to(tl.float32)
    ci=tl.arange(0,128)
    chosen=tl.load(LUT+h*(N//64)*144+(tile*BQ//64)*144+ci,ci<72,-999)
    mi=tl.full((BQ,),-1.e30,tl.float32);den=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,16),tl.float32)
    rr=tl.arange(0,BP*4);rank=tl.arange(0,16);pages=tl.arange(0,BP)
    for first in range(tr.cdiv(P,BP)):
        ids=first*BP+pages
        rep=first*BP*4+rr
        keys=tl.load(PK+h*CAP*4*D+rep[None,:]*D+di[:,None],rep[None,:]<P*4,0)
        logits=tl.dot(q,keys,input_precision='tf32x3')*(1.4426950408889634/math.sqrt(D))
        logits=tl.reshape(logits,(BQ,BP,4))
        mx=tl.max(logits,2);z=mx+tl.log2(tl.sum(tl.exp2(logits-mx[:,:,None]),2))+4.0
        selected=tl.sum((ids[:,None]==chosen[None,:]).to(tl.int32),1)>0
        z=tl.where((ids[None,:]<P)&(~selected[None,:]),z,-float('inf'))
        new=tl.maximum(mi,tl.max(z,1));scale=tl.exp2(mi-new);prob=tl.exp2(z-new[:,None])
        codes=tl.load(CODE+h*CAP*8+ids[:,None]*8+rank[None,:],(ids[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
        acc=acc*scale[:,None]+tl.dot(prob,codes,input_precision='tf32x3')
        den=den*scale+tl.sum(prob,1);mi=new
    result=acc/den[:,None]
    tl.store(OUT+h*N*8+qi[:,None]*8+rank[None,:],result,(qi[:,None]<N)&(rank[None,:]<8))
    tl.store(LSE+h*N+qi,mi+tl.log2(den),qi<N)

@tr.jit
def _merge(EXACT,ELSE,TAIL,TLSE,BASIS,CENTER,GAMMA,OUT,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr):
    tile=tl.program_id(0);h=tl.program_id(1);qi=tile*BQ+tl.arange(0,BQ);di=tl.arange(0,D);ri=tl.arange(0,16)
    code=tl.load(TAIL+h*N*8+qi[:,None]*8+ri[None,:],(qi[:,None]<N)&(ri[None,:]<8),0)
    basis=tl.load(BASIS+h*D*8+di[None,:]*8+ri[:,None],ri[:,None]<8,0).to(tl.float32)
    value=tl.dot(code,basis,input_precision='tf32x3')+tl.load(CENTER+h*D+di)[None,:]
    ls=tl.load(ELSE+h*N+qi,qi<N,0);lt=tl.load(TLSE+h*N+qi,qi<N,0)
    # Existing sparse kernel reports log2-partition.
    rho=1/(1+tl.exp2(ls-lt));gamma=tl.load(GAMMA+h)
    original=tl.load(EXACT+qi[:,None]*H*D+h*D+di[None,:],qi[:,None]<N,0).to(tl.float32)
    result=original+(gamma*rho)[:,None]*(value-original)
    tl.store(OUT+qi[:,None]*H*D+h*D+di[None,:],result,qi[:,None]<N)

def compensate(q,pk,code,lut,exact,lse,basis,center,gamma,p):
    _,n,h,d=q.shape;cap=pk.shape[1]
    tail=torch.empty(h,n,8,device=q.device,dtype=torch.float32);tlse=torch.empty(h,n,device=q.device)
    out=torch.empty_like(exact)
    _tail[(tr.cdiv(n,16),h)](q,pk,code,lut,tail,tlse,p,cap,n,h,d,16,16,num_warps=4)
    _merge[(tr.cdiv(n,16),h)](exact,lse,tail,tlse,basis,center,gamma,out,n,h,d,16,num_warps=4)
    return out,tail,tlse

@torch.no_grad()
def reference(q,pk,code,lut,exact,lse,basis,center,gamma,p):
    _,n,h,d=q.shape;qq=q[0].transpose(0,1).float();key=pk[:,:p].flatten(1,2)
    logits=(qq@key.transpose(1,2)/math.sqrt(d)).reshape(h,n,p,4).logsumexp(-1)+math.log(16)
    selected=torch.zeros(h,n//64,p,device=q.device,dtype=torch.bool).scatter_(-1,lut[0,:,:,:72].long(),True)
    logits=logits.masked_fill(selected.repeat_interleave(64,1),-torch.inf)
    tail=logits.softmax(-1)@code[:,:p].float();lt=logits.logsumexp(-1)/math.log(2)
    y=tail@basis.float().transpose(1,2)+center[:,None]
    x=exact[0].transpose(0,1).float();rho=((lt-lse[0])*math.log(2)).sigmoid()*gamma[:,None]
    out=(x+rho[...,None]*(y-x)).transpose(0,1)[None].to(q.dtype)
    return out,tail,lt
