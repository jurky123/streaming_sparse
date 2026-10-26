import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
from ..opt21.attention import _merge_restore
@tr.jit
def _quant(K,KI,KS,N:tl.constexpr,B:tl.constexpr,START:tl.constexpr,SEG:tl.constexpr):
    PAGES=tr.cdiv(SEG,64)
    page=START+tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,128)
    ki=page//PAGES*SEG+page%PAGES*64+t;valid=(ki<N)&(page%PAGES*64+t<SEG)
    k=tl.load(K+(h*N+ki[:,None])*128+d[None,:],valid[:,None],0.).to(tl.float32)
    scale=tl.maximum(tl.max(tl.max(tl.abs(k),1),0)/127.,1e-10)
    tl.store(KI+(h*N+ki[:,None])*128+d[None,:],nearbyint(k/scale).to(tl.int8),valid[:,None])
    tl.store(KS+h*B+page,scale)

class Quantized:
    def __init__(self,cache):
        self.cache=cache;self.seg=cache.seg;h,n,d=cache.k.shape;self.b=tr.cdiv(n,self.seg)*tr.cdiv(self.seg,64)
        self.k=torch.empty_like(cache.k,dtype=torch.int8)
        self.scale=torch.empty((h,self.b),device=cache.k.device,dtype=torch.float32)
        self.nhpages=cache.nh//self.seg*tr.cdiv(self.seg,64)
        self.history()
    def history(self):
        h,n,d=self.cache.k.shape
        _quant[(self.nhpages,h)](self.cache.k,self.k,self.scale,n,self.b,0,self.seg,num_warps=4)
    def current(self):
        h,n,d=self.cache.k.shape
        _quant[(self.b-self.nhpages,h)](self.cache.k,self.k,self.scale,n,self.b,self.nhpages,self.seg,num_warps=4)

@tr.jit
def _shared(Q,K,V,KS,IDS,O,LSE,N:tl.constexpr,NK:tl.constexpr,B:tl.constexpr,
            G:tl.constexpr,T:tl.constexpr,KEEP:tl.constexpr,SPLITS:tl.constexpr,INT8:tl.constexpr,SEG:tl.constexpr):
    PAGES=tr.cdiv(SEG,64)
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
        ki=page//PAGES*SEG+page%PAGES*64+t
        valid=(ki<NK)&(page%PAGES*64+t<SEG)
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

def attention(q,kv,ids,quant,seg,ix=None,splits=2):
 h,n,d=q.shape;b=kv.state['key'].shape[1];out=torch.empty_like(q)
 partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
 lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
 _shared[(tr.cdiv(n,32),h,splits)](q,quant.k,kv.v,quant.scale,ids,partial,lse,n,kv.k.shape[1],b,32,ids.shape[1],ids.shape[2],splits,True,seg,num_warps=2,num_stages=2)
 _merge_restore[(tr.cdiv(n,8),h)](partial,lse,ix if ix is not None else ids,out,n,splits,8,ix is not None,num_warps=4,enable_fp_fusion=False)
 return out
