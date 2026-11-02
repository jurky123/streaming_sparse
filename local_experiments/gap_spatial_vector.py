"""Parallel token-axis softmax, replacing serial lane loops; changed summation."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _softmax(X,O,T:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);f=tl.program_id(1)*B+tl.arange(0,B);t=tl.arange(0,R)
    x=tl.load(X+(row*T+t[:,None])*32+f[None,:],t[:,None]<T,-float('inf'))
    e=tl.exp(x-tl.max(x,0)[None,:]);y=e/tl.sum(e,0)[None,:]
    tl.store(O+(row*T+t[:,None])*32+f[None,:],y,t[:,None]<T)
def spatial(x,block=2,warps=8):
    assert x.is_contiguous() and x.dtype==torch.float32 and x.shape[-1]==32
    t=x.shape[-2];out=torch.empty_like(x)
    compiled=_softmax[(x.numel()//(t*32),32//block)](x,out,t,tr.next_power_of_2(t),block,num_warps=warps,enable_fp_fusion=False)
    return out
