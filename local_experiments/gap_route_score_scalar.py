import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _score_tiled(PEAK,MASS,CODE,VV,KV,W,MASK,DEN,SEL,ALL,RATIO,STD,S,P:tl.constexpr,B:tl.constexpr,SH:tl.constexpr,ST:tl.constexpr,SP:tl.constexpr):
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

    v0=(tl.full((B,),1.,tl.float32))*tl.load(W+h*14+0)
    v1=(lc)*tl.load(W+h*14+1)
    v2=(peak)*tl.load(W+h*14+2)
    v3=(spread)*tl.load(W+h*14+3)
    v4=(vv)*tl.load(W+h*14+4)
    v5=(kv)*tl.load(W+h*14+5)
    v6=(current)*tl.load(W+h*14+6)
    v7=(current*lc)*tl.load(W+h*14+7)
    v8=(mask)*tl.load(W+h*14+8)
    v9=(tl.load(RATIO+row*P+p,p<P,0))*tl.load(W+h*14+9)
    v10=(tl.minimum(tl.maximum(align,-10.),10.)/5.)*tl.load(W+h*14+10)
    v11=(tl.minimum(mag,10.)/5.)*tl.load(W+h*14+11)
    v12=(tl.minimum(tl.maximum(mass*align,-2.),2.))*tl.load(W+h*14+12)
    v13=(tl.minimum(mass*mag,2.))*tl.load(W+h*14+13)
    linear=(((v0+v1)+(v2+v3))+((v4+v5)+(v6+v7)))+(((v8+v9)+(v10+v11))+(v12+v13))
    tl.store(S+row*P+p,lm+lib.tanh(linear),p<P)
