"""Experimental FP32 route projection using TF32x3; not an exact replacement."""
import triton as tr
import triton.language as tl
from .compensation_stable_kernels import _round_tf32
@tr.jit
def _project(Q,K,O,M:tl.constexpr,N:tl.constexpr,D:tl.constexpr,KH:tl.constexpr,KN:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32,MODE:tl.constexpr=3):
    m=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);h=tl.program_id(2)
    k=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tr.cdiv(D,BK)):
        d=start*BK+k
        q=tl.load(Q+(h*M+m[:,None])*D+d[None,:],(m[:,None]<M)&(d[None,:]<D),0)
        kv=tl.load(K+h*KH+n[None,:]*KN+d[:,None],(n[None,:]<N)&(d[:,None]<D),0)
        if MODE==3:
            acc=tl.dot(q,kv,acc,input_precision='tf32x3')
        else:
            kh=kv.to(tl.bfloat16);kl=(kv-kh.to(tl.float32)).to(tl.bfloat16)
            acc=tl.dot(q.to(tl.bfloat16),kl,acc)
            acc=tl.dot(q.to(tl.bfloat16),kh,acc)
    tl.store(O+(h*M+m[:,None])*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))
def project(q,k,out,bm=32,bn=64,bk=32,mode=3):
    h,m,d=q.shape;_,n,_=k.shape
    assert q.is_contiguous() and k.stride(-1)==1 and out.is_contiguous()
    _project[(tr.cdiv(m,bm),tr.cdiv(n,bn),h)](q,k,out,m,n,d,*k.stride()[:2],bm,bn,bk,mode,num_warps=4,num_stages=2)
    return out
