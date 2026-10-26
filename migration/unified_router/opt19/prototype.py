"""Page-local fused prototype builder, no materialized token assignments.

tf32x3 is the conservative path; fp16 is an explicitly measured approximation.
Projection commutes with the weighted mean, avoiding 128-D prototype storage.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _segment(K, W, BIAS, BASIS, OUT, LC, N:tl.constexpr, NP:tl.constexpr,
             KH:tl.constexpr, KN:tl.constexpr, WH:tl.constexpr,
             BH:tl.constexpr, BD:tl.constexpr, HALF:tl.constexpr):
    page=tl.program_id(0);h=tl.program_id(1)
    t=page*64+tl.arange(0,64);d=tl.arange(0,128);p=tl.arange(0,16);r=tl.arange(0,64)
    k=tl.load(K+h*KH+t[:,None]*KN+d[None,:],t[:,None]<N,0).to(tl.float32)
    w=tl.load(W+h*WH+p[None,:]*128+d[:,None],p[None,:]<NP,0)
    basis=tl.load(BASIS+h*BH+d[:,None]*BD+r[None,:])
    if HALF:
        logits=tl.dot(k.to(tl.float16),w.to(tl.float16))
        kp=tl.dot(k.to(tl.float16),basis.to(tl.float16))
    else:
        logits=tl.dot(k,w,input_precision='tf32x3')
        kp=tl.dot(k,basis,input_precision='tf32x3')
    bias=tl.load(BIAS+h*NP+p,p<NP,0)
    logits=tl.where(p[None,:]<NP,logits+bias[None,:],float('-inf'))
    a=tl.exp(logits-tl.max(logits,1)[:,None])
    a=a/tl.sum(a,1)[:,None]
    a=tl.where(t[:,None]<N,a,0.)
    count=tl.sum(a,0)
    if HALF:
        weighted=tl.dot(tl.trans(a).to(tl.float16),kp.to(tl.float16))
    else:
        weighted=tl.dot(tl.trans(a),kp,input_precision='tf32x3')
    projected=weighted/tl.maximum(count,1e-20)[:,None]
    pages=tr.cdiv(N,64)
    tl.store(OUT+((h*pages+page)*NP+p[:,None])*64+r[None,:],projected,p[:,None]<NP)
    lc=tl.where(count>1e-8,tl.log(tl.maximum(count,1e-30)),float('-inf'))
    tl.store(LC+(h*pages+page)*NP+p,lc,p<NP)


def segment(k,weight,bias,basis,half=False):
    h,n,d=k.shape;p=weight.shape[1];b=tr.cdiv(n,64)
    out=torch.empty((h,b,p,64),device=k.device,dtype=torch.float16)
    lc=torch.empty((h,b,p),device=k.device,dtype=torch.float32)
    _segment[(b,h)](k,weight,bias,basis,out,lc,n,p,k.stride(0),k.stride(1),
                    weight.stride(0),basis.stride(0),basis.stride(1),half,num_warps=4)
    return out.flatten(1,2),lc,None
