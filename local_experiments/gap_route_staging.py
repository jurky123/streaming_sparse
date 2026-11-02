"""Single staging launch with explicit scalar metadata for installed Triton ABI."""
import triton as tr,triton.language as tl
@tr.jit
def _stage(S0,S1,S2,S3,S4,S5,S6,S7,S8,D0,D1,D2,D3,D4,D5,D6,D7,D8,N0:tl.constexpr,N1:tl.constexpr,N2:tl.constexpr,N3:tl.constexpr,N4:tl.constexpr,N5:tl.constexpr,N6:tl.constexpr,N7:tl.constexpr,N8:tl.constexpr,B:tl.constexpr):
    pid=tl.program_id(0)
    if pid < tl.cdiv(N0,B):
        x=(pid-(0))*B+tl.arange(0,B)
        value=tl.load(S0+x,x<N0,0)
        tl.store(D0+x,value,x<N0)
    elif pid < tl.cdiv(N0,B)+tl.cdiv(N1,B):
        x=(pid-(tl.cdiv(N0,B)))*B+tl.arange(0,B)
        value=tl.load(S1+x,x<N1,0)
        tl.store(D1+x,value,x<N1)
    elif pid < tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B):
        x=(pid-(tl.cdiv(N0,B)+tl.cdiv(N1,B)))*B+tl.arange(0,B)
        value=tl.load(S2+x,x<N2,0)
        tl.store(D2+x,value,x<N2)
    elif pid < tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B):
        x=(pid-(tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)))*B+tl.arange(0,B)
        value=tl.load(S3+x,x<N3,0)
        tl.store(D3+x,value,x<N3)
    elif pid < tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B):
        x=(pid-(tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)))*B+tl.arange(0,B)
        value=tl.load(S4+x,x<N4,0)
        tl.store(D4+x,value,x<N4)
    elif pid < tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B)+tl.cdiv(N5,B):
        x=(pid-(tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B)))*B+tl.arange(0,B)
        value=tl.load(S5+x,x<N5,0)
        tl.store(D5+x,value,x<N5)
    elif pid < tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B)+tl.cdiv(N5,B)+tl.cdiv(N6,B):
        x=(pid-(tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B)+tl.cdiv(N5,B)))*B+tl.arange(0,B)
        value=tl.load(S6+x,x<N6,0)
        tl.store(D6+x,value,x<N6)
    elif pid < tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B)+tl.cdiv(N5,B)+tl.cdiv(N6,B)+tl.cdiv(N7,B):
        x=(pid-(tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B)+tl.cdiv(N5,B)+tl.cdiv(N6,B)))*B+tl.arange(0,B)
        value=tl.load(S7+x,x<N7,0)
        tl.store(D7+x,value,x<N7)
    elif pid < tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B)+tl.cdiv(N5,B)+tl.cdiv(N6,B)+tl.cdiv(N7,B)+tl.cdiv(N8,B):
        x=(pid-(tl.cdiv(N0,B)+tl.cdiv(N1,B)+tl.cdiv(N2,B)+tl.cdiv(N3,B)+tl.cdiv(N4,B)+tl.cdiv(N5,B)+tl.cdiv(N6,B)+tl.cdiv(N7,B)))*B+tl.arange(0,B)
        value=tl.load(S8+x,x<N8,0)
        tl.store(D8+x,value,x<N8)
def stage(pairs):
    assert len(pairs)==9
    dst=tuple(a for a,b in pairs);src=tuple(b.contiguous() for a,b in pairs)
    assert all(a.is_contiguous() and a.numel()==b.numel() for a,b in zip(dst,src))
    sizes=tuple(x.numel() for x in src)
    _stage[(sum(tr.cdiv(n,1024) for n in sizes),)](*src,*dst,*sizes,1024,num_warps=4)
