"""Candidate fusion of 32-dimensional power-iteration normalization."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _norm(X,Y,N:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B);d=tl.arange(0,32)
    x=tl.load(X+row[:,None]*32+d[None,:],row[:,None]<N,0)
    v=tl.sum(tl.reshape(x*x,(B,16,2)),2)
    v=tl.sum(tl.reshape(v,(B,8,2)),2)
    v=tl.sum(tl.reshape(v,(B,4,2)),2)
    v=tl.sum(tl.reshape(v,(B,2,2)),2)
    den=tl.maximum(lib.sqrt_rn(tl.sum(v,1)),1.e-8)
    tl.store(Y+row[:,None]*32+d[None,:],tl.div_rn(x,den[:,None]),row[:,None]<N)
def normalize(x):
    assert x.is_contiguous() and x.shape[-1]==32
    out=torch.empty_like(x);n=x.numel()//32
    _norm[(tr.cdiv(n,4),)](x,out,n,4,num_warps=4,enable_fp_fusion=False)
    return out
if __name__=='__main__':
    import json
    from .learned_cache.io import atomic
    torch.manual_seed(123);rows=[]
    for n in [12,36,108,216,432,864,1728]:
        for scale in [1e-9,1.,1e5]:
            x=torch.randn(n,32,device='cuda')*scale;ref=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-8);y=normalize(x)
            rows.append(dict(n=n,scale=scale,exact=torch.equal(y,ref),maxerr=float((y-ref).abs().max())))
    atomic('artifacts/gap_hotspots_20261031/group_norm.json',dict(rows=rows,passed=all(r['exact'] for r in rows)));print(json.dumps(rows))
