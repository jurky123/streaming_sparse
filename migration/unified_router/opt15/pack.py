import triton as tr
import triton.language as tl
@tr.jit
def _pack_union(M,QP,QS,ORDER,MO,QO,SO,IDS,COUNTS,Q:tl.constexpr,B:tl.constexpr,G:tl.constexpr,P:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);oi=g*G+tl.arange(0,G);qi=tl.load(ORDER+h*Q+oi,oi<Q,Q);p=tl.arange(0,P)
 m=tl.load(M+(h*Q+qi[:,None])*B+p[None,:],(qi[:,None]<Q)&(p[None,:]<B),False)
 tl.store(MO+(h*Q+oi[:,None])*B+p[None,:],m,(oi[:,None]<Q)&(p[None,:]<B))
 used=tl.sum(m.to(tl.int32),0)>0;rank=tl.cumsum(used.to(tl.int32),0)-1
 tl.store(IDS+(h*tr.cdiv(Q,G)+g)*B+rank,p,used);tl.store(COUNTS+h*tr.cdiv(Q,G)+g,tl.sum(used.to(tl.int32),0))
 d=tl.arange(0,128);q=tl.load(QP+(h*Q+qi[:,None])*128+d[None,:],qi[:,None]<Q,0)
 tl.store(QO+(h*Q+oi[:,None])*128+d[None,:],q,oi[:,None]<Q)
 scale=tl.load(QS+h*tr.cdiv(Q,32)+qi//32,qi<Q,0);tl.store(SO+h*Q+oi,scale,oi<Q)
