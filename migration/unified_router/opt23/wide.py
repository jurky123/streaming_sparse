"""Same query groups and selected pages; process multiple pages per iteration."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
from ..opt21.attention import _merge_restore
@tr.jit
def _wide(Q,K,V,KS,IDS,O,LSE,N:tl.constexpr,NK:tl.constexpr,B:tl.constexpr,
            G:tl.constexpr,T:tl.constexpr,KEEP:tl.constexpr,SPLITS:tl.constexpr,INT8:tl.constexpr,SEG:tl.constexpr,BN:tl.constexpr):
    PAGES=tr.cdiv(SEG,64)
    tile=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2)
    qi=tile*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,BN)
    q=tl.load(Q+(h*N+qi[:,None])*128+d[None,:],qi[:,None]<N,0)
    if INT8:
        qs=tl.maximum(tl.max(tl.max(tl.abs(q.to(tl.float32)),1),0)/127.,1e-10)
        q=nearbyint(q.to(tl.float32)/qs).to(tl.int8)
    maximum=tl.full((G,),float('-inf'),tl.float32)
    denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
    per=tr.cdiv(KEEP,SPLITS)
    end=tl.minimum((split+1)*per,KEEP)
    for j in range(split*per,end,BN//64):
        page=tl.load(IDS+(h*T+tile)*KEEP+j+t//64,j+t//64<end,0)
        ki=page//PAGES*SEG+page%PAGES*64+t%64
        valid=(ki<NK)&(page%PAGES*64+t%64<SEG)&(j+t//64<end)
        k=tl.load(K+(h*NK+ki[None,:])*128+d[:,None],valid[None,:],0)
        logits=tl.dot(q,k).to(tl.float32)
        if INT8:logits=logits*(qs*tl.load(KS+h*B+page)[None,:])*0.12751743082459617
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


def attention(q,kv,ids,quant,seg,ix,splits=2,bn=128,warps=4,stages=2):
 h,n,d=q.shape;b=kv.state['key'].shape[1];out=torch.empty_like(q)
 partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
 lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
 _wide[(tr.cdiv(n,32),h,splits)](q,quant.k,kv.v,quant.scale,ids,partial,lse,n,kv.k.shape[1],b,32,ids.shape[1],ids.shape[2],splits,True,seg,bn,num_warps=warps,num_stages=stages)
 _merge_restore[(tr.cdiv(n,8),h)](partial,lse,ix,out,n,splits,8,True,num_warps=4,enable_fp_fusion=False)
 return out
