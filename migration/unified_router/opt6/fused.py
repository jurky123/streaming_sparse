"""Keep normalized FP16 inputs, factor the first affine layer by action side.

11 features are query-only, six depend on removal, six on addition, one on
both. The 81 actions and frozen head stay unchanged; accumulation order differs.
"""
import math
import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import erf

@tr.jit
def _fused_factor(ST,X,V,C,O,R,A,W1,B1,MU,SIG,STEP,S,W2,B2,OUT,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,HIST:tl.constexpr,CUR:tl.constexpr,BETA:tl.constexpr,PAIR:tl.constexpr):
    row=tl.program_id(0);head=row//N
    d=tl.arange(0,32);slot=tl.arange(0,16);f=tl.arange(0,16);hidden=tl.arange(0,16)
    center=tl.load(ST+row*80+32+d);av=center-tl.load(ST+row*80+d)
    mean=tl.load(ST+row*80+66);std=tl.load(ST+row*80+67)
    ri=tl.load(R+row*8+slot,slot<8,-1);aj=tl.load(A+row*8+slot,slot<8,-1)
    xi=tl.load(V+(row*16+slot[:,None])*32+d[None,:],(slot[:,None]<8)&(ri[:,None]>=0),0.)
    xj=tl.load(V+(row*16+8+slot[:,None])*32+d[None,:],(slot[:,None]<8)&(aj[:,None]>=0),0.)
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
    if PAIR=='fp16':cross=tl.dot(xi.to(tl.float16),tl.trans(xj.to(tl.float16)))
    else:cross=tl.dot(xi,tl.trans(xj),input_precision=PAIR)
    mu5=tl.load(MU+head*24+5);sig5=tl.load(SIG+head*24+5)
    cross=((cross-mu5)/sig5).to(tl.float16).to(tl.float32)
    y=base[None,None,:]+yi[:,None,:]+yj[None,:,:]+cross[:,:,None]*tl.load(W1+head*384+hidden*24+5).to(tl.float16).to(tl.float32)[None,None,:]
    y=.5*y*(1.+erf(y*.7071067811865476))
    pred=tl.sum(y*tl.load(W2+head*16+hidden)[None,None,:],2)+tl.load(B2+head)
    si=slot[:,None];sj=slot[None,:]
    action=tl.where(si==8,tl.where(sj==8,0,1+sj),tl.where(sj==8,9+si,17+si*8+sj))
    count=tl.load(ST+row*80+71).to(tl.int32)
    valid=((si==8)&(sj==8))|((si==8)&(sj<8)&(aj[None,:]>=0))|((sj==8)&(si<8)&(ri[:,None]>=0)&(count>1))|((si<8)&(sj<8)&(ri[:,None]>=0)&(aj[None,:]>=0))
    pred=tl.where((si==8)&(sj==8),0.,tl.where(valid,pred,float('-inf')))
    best=tl.max(tl.max(pred,1),0)
    chosen=tl.min(tl.min(tl.where(pred==best,action,2147483647),1),0)
    idxi=tl.where(chosen<17,chosen-9,(chosen-17)//8)
    idxj=tl.where(chosen<9,chosen-1,(chosen-17)%8)
    removed=tl.load(R+row*8+idxi,chosen>=9,-1)
    added=tl.load(A+row*8+idxj,((chosen>=1)&(chosen<=8))|(chosen>=17),-1)
    pages=tl.arange(0,P);selected=tl.load(S+row*B+pages,pages<B,False)
    tl.store(OUT+row*B+pages,(selected&(pages!=removed))|(pages==added),pages<B)
