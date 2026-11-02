"""Fuse cheap feature projection, normalization and learned subgroup weights.
Projection centering commutes algebraically with the bank; changed FP order.
"""
import torch,triton as tr,triton.language as tl
from .compensation_stable_kernels import _round_tf32
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _feature_weights(K,V,BANK,WK,WV,WF,O,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr):
    row=tl.program_id(0);h=row//P;p=row%P
    t=tl.arange(0,64);d=tl.arange(0,128);r=tl.arange(0,32)
    k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]).to(tl.float32)
    bank=tl.load(BANK+(h*32+r[None,:])*128+d[:,None])
    bh=_round_tf32(bank);bl=bank-bh
    proj=tl.dot(k,bl,input_precision='tf32');proj=tl.dot(k,bh,proj,input_precision='tf32')
    f=(proj-tl.sum(proj,0)[None,:]/64.)*0.08838834764831843
    norm=lib.sqrt(tl.sum(tl.sum(f*f,1),0)/2048.)
    f=f/tl.maximum(norm,1.e-6)
    wf=tl.load(WF+(h*4+t[:,None]//16)*32+r[None,:])
    zf=tl.sum(f*wf,1)
    v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]).to(tl.float32)
    wk=tl.load(WK+(h*4+t[:,None]//16)*128+d[None,:]);wv=tl.load(WV+(h*4+t[:,None]//16)*128+d[None,:])
    z=(tl.sum(k*wk,1)+tl.sum(v*wv,1))+zf
    z=tl.reshape(z,(4,16));z=z-tl.sum(z,1)[:,None]/16.
    z=2*lib.tanh(z/2);a=tl.exp(z-tl.max(z,1)[:,None]);a=a/tl.sum(a,1)[:,None]
    tl.store(O+row*64+t,tl.reshape(a,(64,)))
def weights(k,v,bank,wk,wv,wf):
    h,p,t,d=k.shape
    assert (t,d)==(64,128) and k.dtype==torch.bfloat16 and v.dtype==torch.bfloat16
    a=torch.empty(h,p,4,16,device=k.device)
    _feature_weights[(h*p,)](k,v,bank,wk,wv,wf,a,p,*k.stride()[:3],*v.stride()[:3],num_warps=4,num_stages=1,enable_fp_fusion=False)
    return a
