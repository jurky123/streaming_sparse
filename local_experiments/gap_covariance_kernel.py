"""Fused page-local rank8 covariance. BF16 Tensor Core projection, FP32 moments.
Changes accumulation/reduction order; validation required. No persistent state.
"""
import torch,triton as tr,triton.language as tl
@tr.jit
def _covariance(K,V,B,O,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0);h=row//P;p=row%P
    t=tl.arange(0,64);d=tl.arange(0,128);r=tl.arange(0,16)
    k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]).to(tl.bfloat16)
    v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]).to(tl.bfloat16)
    basis=tl.load(B+h*BH+d[:,None]*BD+r[None,:]*BR,r[None,:]<8,0).to(tl.bfloat16)
    a=tl.dot(k,basis);b=tl.dot(v,basis)
    a=tl.reshape(a,(4,16,16));b=tl.reshape(b,(4,16,16))
    ac=a-tl.sum(a,1)[:,None,:]/16.;bc=b-tl.sum(b,1)[:,None,:]/16.
    cov=tl.sum(tl.sum(ac*bc,1),0)/64.
    tl.store(O+row*8+r,cov,r<8)
def covariance(k,v,basis):
    h,p,t,d=k.shape;assert (t,d)==(64,128) and basis.shape==(h,128,8)
    assert k.stride(-1)==v.stride(-1)==1
    out=torch.empty(h,p,8,device=k.device,dtype=torch.bfloat16)
    _covariance[(h*p,)](k,v,basis,out,p,*k.stride()[:3],*v.stride()[:3],*basis.stride(),num_warps=4,num_stages=1,enable_fp_fusion=False)
    return out
