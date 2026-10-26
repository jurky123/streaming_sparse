import triton as tr
import triton.language as tl
@tr.jit
def _union(M, IDS, COUNTS, ORDER, Q:tl.constexpr,B:tl.constexpr,G:tl.constexpr,P:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1)
    oi=g*G+tl.arange(0,G);qi=tl.load(ORDER+h*Q+oi,oi<Q,Q);p=tl.arange(0,P)
    m=tl.load(M+(h*Q+qi[:,None])*B+p[None,:],(qi[:,None]<Q)&(p[None,:]<B),False)
    used=tl.sum(m.to(tl.int32),0)>0
    rank=tl.cumsum(used.to(tl.int32),0)-1
    tl.store(IDS+(h*tr.cdiv(Q,G)+g)*B+rank,p,used)
    tl.store(COUNTS+h*tr.cdiv(Q,G)+g,tl.sum(used.to(tl.int32),0))


