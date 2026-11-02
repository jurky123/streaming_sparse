import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _features_tiled(PEAK,MASS,CODE,VV,KV,W,MASK,DEN,SEL,ALL,RATIO,STD,S,P:tl.constexpr,B:tl.constexpr,SH:tl.constexpr,ST:tl.constexpr,SP:tl.constexpr):
    row=tl.program_id(0);h=row//72;t=row%72;p=tl.program_id(1)*B+tl.arange(0,B);d=tl.arange(0,8)
    mass=tl.load(MASS+row*P+p,p<P,0);lm=lib.log(tl.maximum(mass,1.e-12));lc=tl.minimum(tl.maximum(lm,-20.),0.)/5.
    spread=tl.load(STD+row*P+p,p<P,0)
    peak=tl.minimum(tl.maximum(lib.log(tl.maximum(tl.load(PEAK+row*P+p,p<P,0),1.e-12))-lm,0.),10.)/5.
    vv=lib.log1p(tl.maximum(tl.load(VV+h*P+p,p<P,0),0.));kv=tl.load(KV+h*P+p,p<P,0)
    current=(p>=P-72).to(tl.float32);mask=tl.load(MASK+row*P+p,p<P,0).to(tl.float32);den=tl.load(DEN+row)
    sel=tl.load(SEL+row*8+d);missing=tl.load(ALL+row*8+d)-sel
    code=tl.load(CODE+h*SH+t*ST+p[:,None]*SP+d[None,:],p[:,None]<P,0).to(tl.float32);diff=code-sel[None,:]
    energy=tl.maximum(tl.sum(tl.sum(tl.reshape(tl.sum(tl.reshape(sel*sel,(4,2)),1),(2,2)),1),0),.01)
    ax=diff*missing[None,:];mx=diff*diff
    asum=tl.sum(tl.sum(tl.reshape(tl.sum(tl.reshape(ax,(B,4,2)),2),(B,2,2)),2),1)
    msum=tl.sum(tl.sum(tl.reshape(tl.sum(tl.reshape(mx,(B,4,2)),2),(B,2,2)),2),1)
    align=tl.div_rn(asum,energy);mag=tl.div_rn(msum,energy)

    fi=tl.arange(0,16)
    values=tl.full((B,16),0.,tl.float32)
    values=tl.where(fi[None,:]==0,(tl.full((B,),1.,tl.float32))[:,None],values)
    values=tl.where(fi[None,:]==1,(lc)[:,None],values)
    values=tl.where(fi[None,:]==2,(peak)[:,None],values)
    values=tl.where(fi[None,:]==3,(spread)[:,None],values)
    values=tl.where(fi[None,:]==4,(vv)[:,None],values)
    values=tl.where(fi[None,:]==5,(kv)[:,None],values)
    values=tl.where(fi[None,:]==6,(current)[:,None],values)
    values=tl.where(fi[None,:]==7,(current*lc)[:,None],values)
    values=tl.where(fi[None,:]==8,(mask)[:,None],values)
    values=tl.where(fi[None,:]==9,(tl.load(RATIO+row*P+p,p<P,0))[:,None],values)
    values=tl.where(fi[None,:]==10,(tl.minimum(tl.maximum(align,-10.),10.)/5.)[:,None],values)
    values=tl.where(fi[None,:]==11,(tl.minimum(mag,10.)/5.)[:,None],values)
    values=tl.where(fi[None,:]==12,(tl.minimum(tl.maximum(mass*align,-2.),2.))[:,None],values)
    values=tl.where(fi[None,:]==13,(tl.minimum(mass*mag,2.))[:,None],values)
    tl.store(S+(row*P+p[:,None])*14+fi[None,:],values,(p[:,None]<P)&(fi[None,:]<14))

