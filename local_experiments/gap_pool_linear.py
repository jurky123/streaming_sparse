"""Coalesced weighted subgroup pooling with four native-order accumulators."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _weighted(K,V,A,PK,PV,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);h=group//(P*4);p=group//4%P;r=group%4
    d=tl.program_id(1)*B+tl.arange(0,B)
    ka=tl.full((B,),0.,tl.float32);kb=ka;kc=ka;kd=ka
    va=ka;vb=ka;vc=ka;vd=ka
    for j in tl.static_range(4):
        t=r*16+4*j;off=group*16+4*j
        a=tl.load(A+off);b=tl.load(A+off+1);c=tl.load(A+off+2);e=tl.load(A+off+3)
        ki=h*KH+p*KP+t*KT+d;vi=h*VH+p*VP+t*VT+d
        ka=ka+a*tl.load(K+ki).to(tl.float32);kb=kb+b*tl.load(K+ki+KT).to(tl.float32)
        kc=kc+c*tl.load(K+ki+2*KT).to(tl.float32);kd=kd+e*tl.load(K+ki+3*KT).to(tl.float32)
        va=va+a*tl.load(V+vi).to(tl.float32);vb=vb+b*tl.load(V+vi+VT).to(tl.float32)
        vc=vc+c*tl.load(V+vi+2*VT).to(tl.float32);vd=vd+e*tl.load(V+vi+3*VT).to(tl.float32)
    tl.store(PK+group*128+d,((ka+kb)+kc)+kd)
    tl.store(PV+group*128+d,((va+vb)+vc)+vd)
def weighted(k,v,a,block=128,warps=4):
    h,p,t,d=k.shape;assert (t,d)==(64,128) and block in [32,64,128] and k.stride(-1)==v.stride(-1)==1
    pk=torch.empty(h,p,4,d,device=k.device);pv=torch.empty_like(pk)
    _weighted[(h*p*4,tr.cdiv(d,block))](k,v,a,pk,pv,p,*k.stride()[:3],*v.stride()[:3],block,num_warps=warps,enable_fp_fusion=False)
    return pk,pv
