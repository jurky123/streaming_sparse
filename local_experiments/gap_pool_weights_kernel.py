"""Page-local learned weights; subgroup centering cancels in final logit centering.
Algebraically equivalent, changed floating-point reduction order.
"""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _weights(K,V,F,WK,WV,WF,O,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,FH:tl.constexpr,FP:tl.constexpr,FG:tl.constexpr,FT:tl.constexpr,FD:tl.constexpr):
    row=tl.program_id(0);g=row%4;p=row//4%P;h=row//(P*4)
    t=tl.arange(0,16);d=tl.arange(0,128)
    k=tl.load(K+h*KH+p*KP+(g*16+t[:,None])*KT+d[None,:]).to(tl.float32)
    v=tl.load(V+h*VH+p*VP+(g*16+t[:,None])*VT+d[None,:]).to(tl.float32)
    wk=tl.load(WK+(h*4+g)*128+d);wv=tl.load(WV+(h*4+g)*128+d)
    f=tl.load(F+h*FH+p*FP+g*FG+t[:,None]*FT+d[None,:]*FD,d[None,:]<32,0)
    wf=tl.load(WF+(h*4+g)*32+d,d<32,0)
    z=(tl.sum(k*wk[None,:],1)+tl.sum(v*wv[None,:],1))+tl.sum(f*wf[None,:],1)
    z=z-tl.sum(z,0)/16.;z=2*lib.tanh(z/2)
    a=tl.exp(z-tl.max(z,0));a=a/tl.sum(a,0)
    tl.store(O+row*16+t,a)
def weights(k,v,f,wk,wv,wf):
    h,p,_,d=k.shape;assert d==128 and f.shape==(h,p,4,16,32)
    a=torch.empty(h,p,4,16,device=k.device)
    _weights[(h*p*4,)](k,v,f,wk,wv,wf,a,p,*k.stride()[:3],*v.stride()[:3],*f.stride(),num_warps=4,enable_fp_fusion=False)
    return a
