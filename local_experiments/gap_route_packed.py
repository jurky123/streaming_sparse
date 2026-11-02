"""Transient BF16 high/low K packing; preprocessing included in public project."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _split(K,HI,LO,N:tl.constexpr,D:tl.constexpr,TOTAL:tl.constexpr,KH:tl.constexpr,KN:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);h=i//(N*D);n=i//D%N;d=i%D
    k=tl.load(K+h*KH+n*KN+d,i<TOTAL,0)
    hi=k.to(tl.bfloat16);lo=(k-hi.to(tl.float32)).to(tl.bfloat16)
    tl.store(HI+i,hi,i<TOTAL);tl.store(LO+i,lo,i<TOTAL)
@tr.jit
def _project(Q,HI,LO,O,M:tl.constexpr,N:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);h=tl.program_id(2)
    k=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tr.cdiv(D,BK)):
        d=start*BK+k
        q=tl.load(Q+(h*M+m[:,None])*D+d[None,:],(m[:,None]<M)&(d[None,:]<D),0).to(tl.bfloat16)
        kh=tl.load(HI+(h*N+n[None,:])*D+d[:,None],(n[None,:]<N)&(d[:,None]<D),0)
        kl=tl.load(LO+(h*N+n[None,:])*D+d[:,None],(n[None,:]<N)&(d[:,None]<D),0)
        acc=tl.dot(q,kl,acc);acc=tl.dot(q,kh,acc)
    tl.store(O+(h*M+m[:,None])*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))
def pack(k):
    h,n,d=k.shape;hi=torch.empty(h,n,d,device=k.device,dtype=torch.bfloat16);lo=torch.empty_like(hi)
    _split[(tr.cdiv(h*n*d,1024),)](k,hi,lo,n,d,h*n*d,*k.stride()[:2],1024,num_warps=4,enable_fp_fusion=False)
    return hi,lo
def project(q,k,out,bm=64,bn=64,bk=128,mode=2):
    assert mode==2 and q.is_contiguous() and k.stride(-1)==1 and out.is_contiguous()
    h,m,d=q.shape;n=k.shape[1];hi,lo=pack(k)
    _project[(tr.cdiv(m,bm),tr.cdiv(n,bn),h)](q,hi,lo,out,m,n,d,bm,bn,bk,num_warps=4,num_stages=2)
    return out
