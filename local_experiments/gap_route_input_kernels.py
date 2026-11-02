"""Fuse query selection/conversion and representative centering/square."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _query(Q,OFF,O,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,T:tl.constexpr,M:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);d=i%D;m=i//D%M;t=i//(D*M)%T;h=i//(D*M*T)
    token=t*64+tl.load(OFF+m,i<N,0)
    value=tl.load(Q+(token*H+h)*D+d,i<N,0).to(tl.float32)
    tl.store(O+i,value,i<N)
def query(q,offsets):
    _,n,h,d=q.shape;t=n//64;m=offsets.numel();assert q.is_contiguous() and d==128
    out=torch.empty(h,t,m,d,device=q.device)
    _query[(tr.cdiv(out.numel(),256),)](q,offsets,out,out.numel(),h,d,t,m,256,num_warps=4)
    return out
@tr.jit
def _ksquare(K,O,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);off=i//128*512+i%128
    a=tl.load(K+off,i<N,0);b=tl.load(K+off+128,i<N,0);c=tl.load(K+off+256,i<N,0);d=tl.load(K+off+384,i<N,0)
    mean=(((a+b)+c)+d)*.25
    a=a-mean;b=b-mean;c=c-mean;d=d-mean
    tl.store(O+off,a*a,i<N);tl.store(O+off+128,b*b,i<N);tl.store(O+off+256,c*c,i<N);tl.store(O+off+384,d*d,i<N)
def kvar(pk):
    assert pk.is_contiguous() and pk.shape[-2:]==(4,128)
    out=torch.empty_like(pk);n=pk.numel()//4
    _ksquare[(tr.cdiv(n,256),)](pk,out,n,256,num_warps=4,enable_fp_fusion=False)
    return out.mean((2,3)).log1p()
