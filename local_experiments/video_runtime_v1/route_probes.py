"""Gather the 16 chosen query vectors straight into the route graph input."""
import triton
import triton.language as tl


@triton.jit
def _select(Q, OFFSETS, OUT, T:tl.constexpr, SH:tl.constexpr, ST:tl.constexpr, SN:tl.constexpr):
    tile=tl.program_id(0)
    h=tl.program_id(1)
    m=tl.arange(0,16)
    d=tl.arange(0,128)
    source=tl.load(OFFSETS+m).to(tl.int32)
    q=tl.load(Q+h*SH+tile*ST+source[:,None]*SN+d[None,:])
    tl.store(OUT+((h*T+tile)*16+m[:,None])*128+d[None,:],q)


def select(query,offsets,out):
    h,t,n,d=query.shape
    assert n==64 and d==128 and query.stride(-1)==1 and offsets.numel()==16
    _select[(t,h)](query,offsets,out,t,*query.stride()[:3],num_warps=4)
    return out
