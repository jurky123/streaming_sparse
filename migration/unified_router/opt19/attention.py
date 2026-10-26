"""Full-tile FlashAttention loop with fixed selected-page lists.

Reuses the existing INT8 quantization convention and split merge. BF16 QK/PV
is a separately measured path. No per-query mask/union tensor is materialized.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
from ..opt9.segmented_attention import _merge,_split
from ..fast_kernels import _union

@tr.jit
def _quant(K,KI,KS,N:tl.constexpr,B:tl.constexpr,START:tl.constexpr):
    page=START+tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,128)
    ki=page//74*4680+page%74*64+t;valid=(ki<N)&(page%74*64+t<4680)
    k=tl.load(K+(h*N+ki[:,None])*128+d[None,:],valid[:,None],0.).to(tl.float32)
    scale=tl.maximum(tl.max(tl.max(tl.abs(k),1),0)/127.,1e-10)
    tl.store(KI+(h*N+ki[:,None])*128+d[None,:],nearbyint(k/scale).to(tl.int8),valid[:,None])
    tl.store(KS+h*B+page,scale)

class Quantized:
    def __init__(self,cache):
        self.cache=cache;h,n,d=cache.k.shape;self.b=tr.cdiv(n,4680)*74
        self.k=torch.empty_like(cache.k,dtype=torch.int8)
        self.scale=torch.empty((h,self.b),device=cache.k.device,dtype=torch.float32)
        self.nhpages=cache.nh//4680*74
        self.history()
    def history(self):
        h,n,d=self.cache.k.shape
        _quant[(self.nhpages,h)](self.cache.k,self.k,self.scale,n,self.b,0,num_warps=4)
    def current(self):
        h,n,d=self.cache.k.shape
        _quant[(self.b-self.nhpages,h)](self.cache.k,self.k,self.scale,n,self.b,self.nhpages,num_warps=4)

@tr.jit
def _shared(Q,K,V,KS,IDS,O,LSE,N:tl.constexpr,NK:tl.constexpr,B:tl.constexpr,
            G:tl.constexpr,T:tl.constexpr,KEEP:tl.constexpr,SPLITS:tl.constexpr,INT8:tl.constexpr):
    tile=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2)
    qi=tile*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,64)
    q=tl.load(Q+(h*N+qi[:,None])*128+d[None,:],qi[:,None]<N,0)
    if INT8:
        qs=tl.maximum(tl.max(tl.max(tl.abs(q.to(tl.float32)),1),0)/127.,1e-10)
        q=nearbyint(q.to(tl.float32)/qs).to(tl.int8)
    maximum=tl.full((G,),float('-inf'),tl.float32)
    denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
    per=tr.cdiv(KEEP,SPLITS)
    for j in range(split*per,tl.minimum((split+1)*per,KEEP)):
        page=tl.load(IDS+(h*T+tile)*KEEP+j)
        ki=page//74*4680+page%74*64+t
        valid=(ki<NK)&(page%74*64+t<4680)
        k=tl.load(K+(h*NK+ki[None,:])*128+d[:,None],valid[None,:],0)
        logits=tl.dot(q,k).to(tl.float32)
        if INT8:logits=logits*(qs*tl.load(KS+h*B+page))*0.12751743082459617
        else:logits=logits*0.12751743082459617
        logits=tl.where(valid[None,:],logits,float('-inf'))
        newmax=tl.maximum(maximum,tl.max(logits,1))
        alpha=tl.exp2(maximum-newmax);prob=tl.exp2(logits-newmax[:,None])
        v=tl.load(V+(h*NK+ki[:,None])*128+d[None,:],valid[:,None],0)
        acc=acc*alpha[:,None]
        acc=tl.dot(prob.to(v.dtype),v,acc)
        denom=denom*alpha+tl.sum(prob,1);maximum=newmax
    if SPLITS==1:
        tl.store(O+(h*N+qi[:,None])*128+d[None,:],acc/denom[:,None],qi[:,None]<N)
    else:
        tl.store(O+((h*N+qi[:,None])*SPLITS+split)*128+d[None,:],acc/denom[:,None],qi[:,None]<N)
        tl.store(LSE+(h*N+qi)*SPLITS+split,maximum+tl.log2(denom),qi<N)

def attention(q,kv,ids,group=32,quant=None,splits=1,warps=4,stages=2):
    h,n,d=q.shape;b=kv.state['key'].shape[1]
    out=torch.empty_like(q)
    if splits>1:
        partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
        lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
    else:partial=out;lse=out
    k=kv.k if quant is None else quant.k
    scale=kv.k if quant is None else quant.scale
    _shared[(tr.cdiv(n,group),h,splits)](q,k,kv.v,scale,ids,partial,lse,n,k.shape[1],b,
        group,ids.shape[1],ids.shape[2],splits,quant is not None,num_warps=warps,num_stages=stages)
    if splits>1:_merge[(h*n,)](partial,lse,out,n,splits,num_warps=4)
    return out


def native(q,kv,mask,quant,group=32):
    """Original per-query kernel, with owned prequantized K and contiguous V."""
    h,n,d=q.shape;b=mask.shape[-1];nh=kv.history_tokens;nk=kv.k.shape[1];splits=2
    ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32)
    counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
    _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
    out=torch.empty_like(q);partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
    lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
    k,v=quant.k,kv.v
    _split[(tr.cdiv(n,group),h,splits)](q,k,v,v,quant.scale,mask,ids,counts,partial,lse,n,nk,b,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),v.stride(0),v.stride(1),nh,group,splits,False,True,
        num_warps=2,num_stages=2)
    _merge[(h*n,)](partial,lse,out,n,splits,num_warps=4)
    return out
