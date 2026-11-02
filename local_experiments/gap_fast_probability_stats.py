"""Native-order probe mean/max and Welford log-spread without large temporaries."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _prob_stats(X,M,PEAK,SPREAD,P:tl.constexpr,N:tl.constexpr,B:tl.constexpr,FMA:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);row=i//P;p=i%P
    means=tl.full((B,2),0.,tl.float32);m2=tl.full((B,2),0.,tl.float32);a=tl.arange(0,2)
    peak=tl.full((B,),0.,tl.float32)
    s0=tl.full((B,),0.,tl.float32);s1=s0;s2=s0;s3=s0
    for j in tl.static_range(8):
        probe=2*j+a
        x=tl.load(X+(row[:,None]*16+probe[None,:])*P+p[:,None],i[:,None]<N,0)
        even,odd=tl.split(x)
        peak=tl.maximum(peak,tl.maximum(even,odd))
        if j%2==0:s0=s0+even;s1=s1+odd
        else:s2=s2+even;s3=s3+odd
        v=tl.log(tl.maximum(x,1.e-12));delta=v-means
        means=means+tl.div_rn(delta,j+1.)
        nd=v-means
        if FMA:m2=tl.fma(delta,nd,m2)
        else:m2=m2+delta*nd
    ma,mb=tl.split(means);va,vb=tl.split(m2);delta=mb-ma
    variance=tl.fma(delta*delta*8.,.5,va+vb) if FMA else (va+vb)+(delta*delta*8.)*.5
    std=tl.sqrt(variance/16.)
    tl.store(M+i,(((s0+s1)+s2)+s3)/16.,i<N)
    tl.store(PEAK+i,peak,i<N)
    tl.store(SPREAD+i,tl.minimum(tl.maximum(std,0.),5.)/5.,i<N)
def probability_stats(prob,block=128,fma=True):
    h,t,n,p=prob.shape;assert n==16 and prob.is_contiguous()
    mass=torch.empty(h,t,p,device=prob.device);peak=torch.empty_like(mass);spread=torch.empty_like(mass)
    _prob_stats[(tr.cdiv(mass.numel(),block),)](prob,mass,peak,spread,p,mass.numel(),block,fma,num_warps=4,enable_fp_fusion=False)
    return mass,peak,spread
