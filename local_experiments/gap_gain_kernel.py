"""Fuse residual gain elementwise work; retain native page reduction and GEMM."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _contrib(Q,M,C,MASK,O,E,P:tl.constexpr,T:tl.constexpr,N:tl.constexpr,QH:tl.constexpr,QT:tl.constexpr,QP:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);d=tl.arange(0,8)
    h=i//(T*P);t=i//P%T;p=i%P
    mass=tl.load(M+i,i<N,0);mask=tl.load(MASK+i,i<N,0)
    q=tl.load(Q+h[:,None]*QH+t[:,None]*QT+p[:,None]*QP+d[None,:],i[:,None]<N,0)
    cov=tl.load(C+(h[:,None]*P+p[:,None])*8+d[None,:],i[:,None]<N,0).to(tl.float32)
    qbar=tl.div_rn(q,16*tl.maximum(mass[:,None],1.e-12))
    contribution=mass[:,None]*(qbar*cov)
    tl.store(O+i[:,None]*8+d[None,:],contribution,i[:,None]<N)
    tl.store(E+i[:,None]*8+d[None,:],contribution*(~mask[:,None]).to(tl.float32),i[:,None]<N)
@tr.jit
def _sum8(x,B:tl.constexpr):
    a=tl.sum(tl.reshape(x,(B,4,2)),2)
    a=tl.sum(tl.reshape(a,(B,2,2)),2)
    return tl.sum(a,1)
@tr.jit
def _gain(C,E,MASK,S,G,O,P:tl.constexpr,T:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);d=tl.arange(0,8);row=i//P;h=row//T
    e=tl.load(E+row[:,None]*8+d[None,:],i[:,None]<N,0)
    c=tl.load(C+i[:,None]*8+d[None,:],i[:,None]<N,0)
    mask=tl.load(MASK+i,i<N,0).to(tl.float32)
    energy=tl.maximum(_sum8(e*e,B),1.e-8)
    gain=tl.div_rn(2*_sum8(e*c,B)+(2*mask-1)*_sum8(c*c,B),energy)
    s=tl.load(S+i,i<N,0);gate=lib.tanh(tl.load(G+h,i<N,0))
    tl.store(O+i,s+gate*tl.minimum(tl.maximum(gain,-5.),5.),i<N)
def contribution(q,mass,cov,mask):
    h,t,p=mass.shape;o=torch.empty(h,t,p,8,device=q.device);e=torch.empty_like(o)
    _contrib[(tr.cdiv(h*t*p,128),)](q,mass,cov,mask,o,e,p,t,h*t*p,*q.stride()[:3],128,num_warps=4,enable_fp_fusion=False)
    return o,e.sum(2)
def finish(c,error,mask,s,gate):
    h,t,p=s.shape;out=torch.empty_like(s)
    _gain[(tr.cdiv(h*t*p,128),)](c,error,mask,s,gate,out,p,t,h*t*p,128,num_warps=4,enable_fp_fusion=False)
    return out
