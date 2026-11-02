"""Experimental FP32 route projection using TF32x3; not an exact replacement."""
import triton as tr
import triton.language as tl
from .compensation_stable_kernels import _round_tf32
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _project_stats(Q,K,Z,W,M:tl.constexpr,N:tl.constexpr,D:tl.constexpr,KH:tl.constexpr,KN:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32,MODE:tl.constexpr=3):
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
    x=tl.reshape(acc*0.08838834764831843,(BM,BN//4,4))
    even,odd=tl.split(tl.reshape(x,(BM,BN//4,2,2)))
    a,c=tl.split(even);b,d=tl.split(odd)
    mx=tl.maximum(tl.maximum(a,b),tl.maximum(c,d))
    ea=tl.exp(a-mx);eb=tl.exp(b-mx);ec=tl.exp(c-mx);ed=tl.exp(d-mx)
    page=tl.program_id(1)*(BN//4)+tl.arange(0,BN//4)
    tl.store(Z+(h*M+m[:,None])*(N//4)+page[None,:],(tl.log((ea+eb)+(ec+ed))+mx)+2.772588722239781,(m[:,None]<M)&(page[None,:]<N//4))
    denom=(ea+ec)+(eb+ed)
    inv=1./denom;ea=ea*inv;eb=eb*inv;ec=ec*inv;ed=ed*inv
    values=tl.reshape(tl.join(tl.join(ea,ec),tl.join(eb,ed)),(BM,BN))
    tl.store(W+(h*M+m[:,None])*N+n[None,:],values,(m[:,None]<M)&(n[None,:]<N))
def project_stats(q,k,z,w):
    h,m,d=q.shape;_,n,_=k.shape
    assert q.is_contiguous() and k.stride(-1)==1 and z.is_contiguous() and w.is_contiguous() and n%4==0
    return _project_stats[(tr.cdiv(m,64),tr.cdiv(n,64),h)](q,k,z,w,m,n,d,*k.stride()[:2],64,64,128,2,num_warps=4,num_stages=2,enable_fp_fusion=False)
