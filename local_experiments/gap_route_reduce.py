"""Fuse selected mass and conditional-V moments; changed reduction tree."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _reduce(MASK,MASS,V,DEN,SEL,ALL,P:tl.constexpr,T:tl.constexpr,SH:tl.constexpr,ST:tl.constexpr,SP:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);h=row//T;t=row%T;p=tl.arange(0,B);d=tl.arange(0,8)
    mass=tl.load(MASS+row*P+p,p<P,0);mask=tl.load(MASK+row*P+p,p<P,0)
    v=tl.load(V+h*SH+t*ST+p[:,None]*SP+d[None,:],p[:,None]<P,0)
    weighted=mass*mask.to(tl.float32);den=tl.maximum(tl.sum(weighted,0),1.e-12)
    selected=tl.div_rn(tl.sum(weighted[:,None]*v,0),den)
    allvalue=tl.sum(mass[:,None]*v,0)
    tl.store(DEN+row,den);tl.store(SEL+row*8+d,selected);tl.store(ALL+row*8+d,allvalue)
def moments(mass,mask,v):
    h,t,p=mass.shape
    den=torch.empty(h,t,1,device=mass.device);sel=torch.empty(h,t,8,device=mass.device);allvalue=torch.empty_like(sel)
    _reduce[(h*t,)](mask,mass,v,den,sel,allvalue,p,t,*v.stride()[:3],tr.next_power_of_2(p),num_warps=4,enable_fp_fusion=False)
    return den,sel,allvalue
