"""Parallelize independent feature columns; preserve spatial softmax lane order."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _softmax(X,O,T:tl.constexpr,F:tl.constexpr,B:tl.constexpr,MUL:tl.constexpr,EXP:tl.constexpr):
    row=tl.program_id(0);f=tl.program_id(1)*B+tl.arange(0,B);lane=tl.arange(0,32)
    mx=tl.full((32,B),-float('inf'),tl.float32)
    for j in range(tr.cdiv(T,32)):
        t=j*32+lane;x=tl.load(X+(row*T+t[:,None])*F+f[None,:],t[:,None]<T,-float('inf'));mx=tl.maximum(mx,x)
    maximum=tl.max(mx,0)
    total=tl.full((32,B),0.,tl.float32)
    for j in range(tr.cdiv(T,32)):
        t=j*32+lane;x=tl.load(X+(row*T+t[:,None])*F+f[None,:],t[:,None]<T,-float('inf'))
        if EXP==0:e=lib.exp(x-maximum[None,:])
        else:e=tl.exp(x-maximum[None,:])
        total=total+e
    # Explicit descending lane tree: Triton otherwise changes order with feature tiling.
    a0,b0=tl.split(tl.permute(tl.reshape(total,(2,16,B)),(1,2,0)))
    a1,b1=tl.split(tl.permute(tl.reshape(a0+b0,(2,8,B)),(1,2,0)))
    a2,b2=tl.split(tl.permute(tl.reshape(a1+b1,(2,4,B)),(1,2,0)))
    a3,b3=tl.split(tl.permute(tl.reshape(a2+b2,(2,2,B)),(1,2,0)))
    a4,b4=tl.split(tl.permute(tl.reshape(a3+b3,(2,1,B)),(1,2,0)))
    den=tl.reshape(a4+b4,(B,))
    for j in range(tr.cdiv(T,32)):
        t=j*32+lane;x=tl.load(X+(row*T+t[:,None])*F+f[None,:],t[:,None]<T,-float('inf'))
        if EXP==0:e=lib.exp(x-maximum[None,:])
        else:e=tl.exp(x-maximum[None,:])
        if MUL:y=e*tl.div_rn(1.,den[None,:])
        else:y=tl.div_rn(e,den[None,:])
        tl.store(O+(row*T+t[:,None])*F+f[None,:],y,t[:,None]<T)
def spatial(x,block=4,mul=False,exp=0):
    assert x.is_contiguous() and x.dtype==torch.float32 and x.shape[-1]==32
    t=x.shape[-2];out=torch.empty_like(x)
    _softmax[(x.numel()//(t*32),32//block)](x,out,t,32,block,mul,exp,num_warps=4,enable_fp_fusion=False)
    return out
