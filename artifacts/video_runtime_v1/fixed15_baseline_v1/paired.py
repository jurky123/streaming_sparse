"""Eight warps, two independent Q64 lists and one Q128 representative tile."""
import math
import triton as tr
import triton.language as tl
from local_experiments.review_v1.tuned_attention import _av

@tr.jit
def _page(q,K,V,ids,h,H:tl.constexpr,D:tl.constexpr):
    n=tl.arange(0,64)
    d=tl.arange(0,D)
    token=ids[:,None]*64+n[None,:]
    key=tl.load(K+(token[:,None,:]*H+h)*D+d[None,:,None])
    val=tl.load(V+(token[:,:,None]*H+h)*D+d[None,None,:])
    score=tl.dot(q,key)
    score=tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;",constraints="=f,f,f",args=[score,(128**-.5)*1.4426950408889634],dtype=tl.float32,is_pure=True,pack=1)
    return score,val

@tr.jit
def paired(Q,KEY,MEAN,DETAIL,BASIS,OUT,K,V,IDS,P:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,EXACT_COUNT:tl.constexpr,hp:tl.constexpr):
    BQ:tl.constexpr=128
    BR:tl.constexpr=64
    t=tl.program_id(0);h=tl.program_id(1)
    qi=t*128+tl.arange(0,128);d=tl.arange(0,D)
    r=tl.arange(0,BR);pages=tl.arange(0,BR//4);rank=tl.arange(0,16)
    q=tl.load(Q+(qi[:,None]*H+h)*D+d[None,:])
    qq=tl.reshape(q,(2,64,128))
    b=tl.arange(0,2)
    lists=IDS+((h*(N//64)+t*2+b)*EXACT_COUNT)
    z0,v0=_page(qq,K,V,tl.load(lists),h,H,D)
    a=tl.max(z0,2)
    prob=tl.exp2(z0-a[:,:,None])
    acc=_av(prob,v0)
    cols=prob
    for i in range(1,EXACT_COUNT):
        z0,v0=_page(qq,K,V,tl.load(lists+i),h,H,D)
        prob=tl.exp2(z0-a[:,:,None])
        acc+=_av(prob,v0)
        cols+=prob
    mass=tl.sum(cols,2)
    exact3=acc/mass[:,:,None]
    le2=a+tl.log2(mass)
    bad=tl.min(((mass>=1.)&(mass<1.21e24)).to(tl.int32),1)==0
    if tl.max(bad.to(tl.int32),0):
        om=tl.full((2,64),-float('inf'),tl.float32)
        den2=tl.full((2,64),0.,tl.float32)
        acc2=tl.full((2,64,128),0.,tl.float32)
        for i in range(EXACT_COUNT):
            z0,v0=_page(qq,K,V,tl.load(lists+i),h,H,D)
            nm2=tl.maximum(om,tl.max(z0,2))
            prob=tl.exp2(z0-nm2[:,:,None]);fade2=tl.exp2(om-nm2)
            acc2=tl.fma(acc2,fade2[:,:,None],_av(prob,v0))
            den2=tl.fma(den2,fade2,tl.sum(prob,2))
            om=nm2
        exact3=tl.where(bad[:,None,None],acc2/den2[:,:,None],exact3)
        le2=tl.where(bad[:,None],om+tl.log2(den2),le2)
    exact=tl.reshape(exact3,(128,128));le=tl.reshape(le2,(128,))
    slots=tl.arange(0,tr.next_power_of_2(EXACT_COUNT))
    ids=tl.load(lists[:,None]+slots[None,:],slots[None,:]<EXACT_COUNT,0)
    bit=tl.where(slots[None,:]<EXACT_COUNT,1<<(ids%32),0).to(tl.uint32)
    words=tl.sum(tl.where(tl.arange(0,16)[None,:,None]==ids[:,None,:]//32,bit[:,None,:],0),2)
    mx=tl.full((128,),-1.e30,tl.float32);den=tl.zeros((128,),tl.float32)
    am=tl.zeros((128,D),tl.float32);ac=tl.zeros((128,16),tl.float32)
    for first in range(tr.cdiv(P*4,BR)):
        rep=first*BR+r;page=first*(BR//4)+pages
        word=tl.sum(tl.where(tl.arange(0,16)[None,:]==first//2,words,0),1)
        omitted=((word[:,None]>>(rep[None,:]//4%32))&1)==0
        keep=tl.reshape(tl.broadcast_to(omitted[:,None,:],(2,64,64)),(128,64))
        key=tl.load(KEY+(h*hp*4+rep[None,:])*D+d[:,None],rep[None,:]<P*4,0)
        z=tl.dot(q,key)*(1.4426950408889634/math.sqrt(D))
        z=tl.where((rep[None,:]<P*4)&keep,z,-float('inf'))
        nm=tl.maximum(mx,tl.max(z,1));fade=tl.exp2(mx-nm);w=tl.exp2(z-nm[:,None]);ww=tl.reshape(w,(BQ,BR//4,4));rr=tl.arange(0,4)
        alpha=tl.sum(ww,2)
        c0=tl.sum(ww*tl.where(rr<2,1.,-1.)[None,None,:],2)
        c1=tl.sum(ww*tl.where(rr%2==0,1.,-1.)[None,None,:],2)
        c2=tl.sum(ww*tl.where((rr==0)|(rr==3),1.,-1.)[None,None,:],2)
        mean=tl.load(MEAN+(h*hp+page[:,None])*D+d[None,:],page[:,None]<P,0).to(tl.float32)
        x=tl.load(DETAIL+((h*hp+page[:,None])*3)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
        y=tl.load(DETAIL+((h*hp+page[:,None])*3+1)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
        v=tl.load(DETAIL+((h*hp+page[:,None])*3+2)*8+rank[None,:],(page[:,None]<P)&(rank[None,:]<8),0).to(tl.float32)
        am=am*fade[:,None]+_av(alpha,mean)
        ac=ac*fade[:,None]+_av(c0,x)+_av(c1,y)+_av(c2,v)
        den=den*fade+tl.sum(w,1);mx=nm
    safe=tl.maximum(den,1.e-30)
    code=tl.where(rank[None,:]<8,ac/safe[:,None],0.)
    basis=tl.load(BASIS+(h*D+d[None,:])*8+rank[:,None],rank[:,None]<8,0).to(tl.float32)
    tail=am/safe[:,None]+tl.dot(code,basis,input_precision='tf32x3')
    lt=tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf'))
    rho=1./(1.+tl.exp2(le-lt))
    tl.store(OUT+(qi[:,None]*H+h)*D+d[None,:],exact*(1.-rho[:,None])+tail*rho[:,None])
