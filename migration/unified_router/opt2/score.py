"""Fused per-page logsumexp/content with direct output layout (no content cat)."""
import math
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _content(L,LC,V,S,C,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,TOTAL:tl.constexpr,OFFSET:tl.constexpr,
             Q:tl.constexpr,PP:tl.constexpr,HAS_VALUE:tl.constexpr):
    qi=tl.program_id(0)*Q+tl.arange(0,Q);page=tl.program_id(1);head=tl.program_id(2)
    p=tl.arange(0,PP);d=tl.arange(0,32)
    logits=tl.load(L+((head*N+qi[:,None])*B+page)*P+p[None,:],(qi[:,None]<N)&(p[None,:]<P),0.)
    logcount=tl.load(LC+(head*B+page)*P+p,p<P,float('-inf'))
    logits=logits/math.sqrt(128)+logcount[None,:]
    maximum=tl.max(logits,1);w=tl.exp(logits-maximum[:,None]);denom=tl.sum(w,1)
    score=maximum+tl.log(denom)
    tl.store(S+(head*N+qi)*TOTAL+OFFSET+page,score,qi<N)
    if HAS_VALUE:
        values=tl.load(V+((head*B+page)*P+p[:,None])*32+d[None,:],p[:,None]<P,0.)
        content=tl.dot(w/denom[:,None],values,input_precision='ieee')
        tl.store(C+((head*N+qi[:,None])*TOTAL+OFFSET+page)*32+d[None,:],content,qi[:,None]<N)

def direct_score(q,basis,projected,tile=32):
    h,n,_=q.shape;total=sum(lc.shape[1]for k,lc,v in projected)
    score=torch.empty((h,n,total),device=q.device,dtype=torch.float32)
    has_value=projected[0][2]is not None
    content=torch.empty((h,n,total,32),device=q.device,dtype=torch.float32)if has_value else None
    qq=torch.bmm(q.float(),basis);offset=0
    for key,lc,value in projected:
        b,p=lc.shape[1:];logits=torch.bmm(qq,key.transpose(1,2))
        _content[(tr.cdiv(n,tile),b,h)](logits,lc,value,score,content,n,b,p,total,offset,tile,max(16,p),has_value,
            num_warps=4,enable_fp_fusion=False)
        offset+=b
    return score,content
