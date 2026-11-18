"""Keep a complete small node in registers throughout its power iteration."""
import torch
import triton
import triton.language as tl

@triton.jit
def _scores(X, O, N:tl.constexpr):
    group = tl.program_id(0)
    t = tl.arange(0,N)
    d = tl.arange(0,32)
    x = tl.load(X + group*N*32 + t[:,None]*32 + d[None,:])
    xc = x - tl.sum(x,0)[None,:]/N
    variance = tl.sum(xc*xc,0)/N
    direction = (d == tl.argmax(variance,0)).to(tl.float32)
    for _ in tl.static_range(4):
        projection = tl.sum(xc*direction[None,:],1)
        direction = tl.sum(xc*projection[:,None],0)
        direction = direction / tl.maximum(tl.sqrt(tl.sum(direction*direction,0)),1e-8)
    tl.store(O + group*N + t, tl.sum(xc*direction[None,:],1))

def scores(x):
    groups, n, dim = x.shape
    assert dim == 32 and x.is_contiguous() and n in (32,64,128,256,512)
    out = torch.empty((groups,n),device=x.device,dtype=x.dtype)
    _scores[(groups,)](x,out,n,num_warps=8,enable_fp_fusion=False)
    return out
