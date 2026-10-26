"""Keep normalized FP16 inputs, factor the first affine layer by action side.

11 features are query-only, six depend on removal, six on addition, one on
both. The 81 actions and frozen head stay unchanged; accumulation order differs.
"""
import math
import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import erf

@tr.jit
def _factor_prepare(ST,X,V,WT,START,COUNT,C,O,R,A,W1,B1,MU,SIG,STEP,T,N:tl.constexpr,B:tl.constexpr,HIST:tl.constexpr,CUR:tl.constexpr,BETA:tl.constexpr,PAIR:tl.constexpr,TOTALR:tl.constexpr):
    row=tl.program_id(0);head=row//N
    d=tl.arange(0,32);slot=tl.arange(0,16);f=tl.arange(0,16);hidden=tl.arange(0,16)
    center=tl.load(ST+row*80+32+d);av=center-tl.load(ST+row*80+d)
    mean=tl.load(ST+row*80+66);std=tl.load(ST+row*80+67)
    ri=tl.load(R+row*8+slot,slot<8,-1);aj=tl.load(A+row*8+slot,slot<8,-1)
    page=tl.where(slot<8,tl.load(R+row*8+slot,slot<8,-1),tl.load(A+row*8+slot-8,slot>=8,-1))
    valid_page=page>=0;proto=tl.arange(0,16)
    start=tl.load(START+page,valid_page,0);count=tl.load(COUNT+page,valid_page,0);pcost=tl.load(C+page,valid_page,1.)
    ww=tl.load(WT+(head*TOTALR+start[:,None]+proto[None,:])*N+row%N,valid_page[:,None]&(proto[None,:]<count[:,None]),0.).to(tl.float32)/pcost[:,None]
    vv=tl.load(V+(head*TOTALR+start[:,None,None]+proto[None,:,None])*32+d[None,None,:],valid_page[:,None,None]&(proto[None,:,None]<count[:,None,None]),0.)
    content=tl.sum(ww[:,:,None]*vv,1)
    left,right=tl.split(tl.permute(content.reshape((2,8,32)),(1,2,0)))
    xi=tl.permute(tl.join(left,tl.full((8,32),0.,tl.float32)),(2,0,1)).reshape((16,32))
    xj=tl.permute(tl.join(right,tl.full((8,32),0.,tl.float32)),(2,0,1)).reshape((16,32))
    xi=tl.where(ri[:,None]>=0,xi-center[None,:],0.)
    xj=tl.where(aj[:,None]>=0,xj-center[None,:],0.)
    ni=tl.sum(xi*xi,1);nj=tl.sum(xj*xj,1)
    pi=tl.sum(xi*av[None,:],1);pj=tl.sum(xj*av[None,:],1)
    zi=tl.where(ri>=0,(tl.load(X+row*B+ri,ri>=0,0.)-mean)/std,0.)
    zj=tl.where(aj>=0,(tl.load(X+row*B+aj,aj>=0,0.)-mean)/std,0.)
    ci=tl.load(C+ri,ri>=0,0.)/64.;cj=tl.load(C+aj,aj>=0,0.)/64.
    oi=tl.load(O+ri,ri>=0,0.);oj=tl.load(O+aj,aj>=0,0.)
    fi=tl.where(f[None,:]==0,ni[:,None],0.)
    fi=tl.where(f[None,:]==1,pi[:,None],fi);fi=tl.where(f[None,:]==2,zi[:,None],fi)
    fi=tl.where(f[None,:]==3,ci[:,None],fi);fi=tl.where(f[None,:]==4,(ri[:,None]>=0).to(tl.float32),fi)
    fi=tl.where(f[None,:]==5,oi[:,None],fi)
    fj=tl.where(f[None,:]==0,nj[:,None],0.)
    fj=tl.where(f[None,:]==1,pj[:,None],fj);fj=tl.where(f[None,:]==2,zj[:,None],fj)
    fj=tl.where(f[None,:]==3,cj[:,None],fj);fj=tl.where(f[None,:]==4,(aj[:,None]>=0).to(tl.float32),fj)
    fj=tl.where(f[None,:]==5,oj[:,None],fj)
    ix=tl.where(f==0,1,tl.where(f==1,3,tl.where(f==2,8,tl.where(f==3,17,tl.where(f==4,19,21)))))
    jx=ix+1
    mui=tl.load(MU+head*24+ix,f<6,0.);sigi=tl.load(SIG+head*24+ix,f<6,1.)
    muj=tl.load(MU+head*24+jx,f<6,0.);sigj=tl.load(SIG+head*24+jx,f<6,1.)
    fi=tl.where(f[None,:]<6,(fi-mui[None,:])/sigi[None,:],0.).to(tl.float16)
    fj=tl.where(f[None,:]<6,(fj-muj[None,:])/sigj[None,:],0.).to(tl.float16)
    wi=tl.load(W1+head*384+hidden[None,:]*24+ix[:,None],f[:,None]<6,0.).to(tl.float16)
    wj=tl.load(W1+head*384+hidden[None,:]*24+jx[:,None],f[:,None]<6,0.).to(tl.float16)
    yi=tl.dot(fi,wi);yj=tl.dot(fj,wj)
    tl.store(T+row*384+16+slot[:,None]*16+hidden[None,:],yi,slot[:,None]<9)
    tl.store(T+row*384+160+slot[:,None]*16+hidden[None,:],yj,slot[:,None]<9)
    # Shared contribution excludes all thirteen action-dependent features.
    shared=tl.where(d==0,tl.sum(av*av,0),0.)
    shared=tl.where(d==6,tl.load(ST+row*80+64),shared);shared=tl.where(d==7,tl.load(ST+row*80+65),shared)
    shared=tl.where(d==10,tl.log(std),shared);shared=tl.where(d==11,BETA,shared)
    shared=tl.where(d==12,tl.load(ST+row*80+68)/tl.load(ST+row*80+69),shared)
    shared=tl.where(d==13,tl.load(ST+row*80+70)/CUR,shared)
    shared=tl.where(d==14,math.log1p(HIST),shared);shared=tl.where(d==15,math.log1p(CUR),shared)
    shared=tl.where(d==16,tl.load(STEP)/3.,shared);shared=tl.where(d==23,tl.load(ST+row*80+72),shared)
    common=(d==0)|(d==6)|(d==7)|((d>=10)&(d<=16))|(d==23)
    mu=tl.load(MU+head*24+d,d<24,0.);sig=tl.load(SIG+head*24+d,d<24,1.)
    norm=tl.where(common,(shared-mu)/sig,0.).to(tl.float16).to(tl.float32)
    w=tl.load(W1+head*384+hidden[None,:]*24+d[:,None],d[:,None]<24,0.).to(tl.float16).to(tl.float32)
    base=tl.sum(norm[:,None]*w,0)+tl.load(B1+head*16+hidden)
    tl.store(T+row*384+hidden,base)
    if PAIR=='fp16':cross=tl.dot(xi.to(tl.float16),tl.trans(xj.to(tl.float16)))
    else:cross=tl.dot(xi,tl.trans(xj),input_precision=PAIR)
    mu5=tl.load(MU+head*24+5);sig5=tl.load(SIG+head*24+5)
    cross=((cross-mu5)/sig5).to(tl.float16).to(tl.float32)
    tl.store(T+row*384+304+slot[:,None]*8+hidden[None,:],cross,(slot[:,None]<8)&(hidden[None,:]<8))
    tl.store(T+row*384+368,(-mu5/sig5).to(tl.float16).to(tl.float32))

