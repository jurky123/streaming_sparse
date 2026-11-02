"""Fuse page-local weighted K/V reductions, preserving native mean16 order."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _weighted(K,V,A,PK,PV,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);h=group//(P*4);p=group//4%P;r=group%4
    d=tl.program_id(1)*B+tl.arange(0,B);g=tl.arange(0,4)
    ka=tl.full((4,B),0.,tl.float32);va=tl.full((4,B),0.,tl.float32)
    for j in tl.static_range(4):
        t=g+4*j
        a=tl.load(A+group*16+t)
        k=tl.load(K+h*KH+p*KP+(r*16+t[:,None])*KT+d[None,:]).to(tl.float32)
        v=tl.load(V+h*VH+p*VP+(r*16+t[:,None])*VT+d[None,:]).to(tl.float32)
        ka=ka+a[:,None]*k;va=va+a[:,None]*v
    k0=tl.sum(tl.where(g[:,None]==0,ka,0.),0);k1=tl.sum(tl.where(g[:,None]==1,ka,0.),0)
    k2=tl.sum(tl.where(g[:,None]==2,ka,0.),0);k3=tl.sum(tl.where(g[:,None]==3,ka,0.),0)
    v0=tl.sum(tl.where(g[:,None]==0,va,0.),0);v1=tl.sum(tl.where(g[:,None]==1,va,0.),0)
    v2=tl.sum(tl.where(g[:,None]==2,va,0.),0);v3=tl.sum(tl.where(g[:,None]==3,va,0.),0)
    tl.store(PK+group*128+d,((k0+k1)+k2)+k3)
    tl.store(PV+group*128+d,((v0+v1)+v2)+v3)
def weighted(k,v,a,block=32):
    h,p,t,d=k.shape;assert (t,d)==(64,128) and a.is_contiguous() and k.stride(-1)==v.stride(-1)==1
    pk=torch.empty(h,p,4,d,device=k.device);pv=torch.empty_like(pk)
    _weighted[(h*p*4,tr.cdiv(d,block))](k,v,a,pk,pv,p,*k.stride()[:3],*v.stride()[:3],block,num_warps=4,enable_fp_fusion=False)
    return pk,pv
