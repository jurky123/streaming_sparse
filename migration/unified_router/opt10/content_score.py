import math
import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint
@tr.jit
def _score_content(Q,K,LC,G,V,S,E,VS,N:tl.constexpr,B:tl.constexpr,
                  QM:tl.constexpr,GP:tl.constexpr,CONTENT_ENERGY:tl.constexpr,
                  W=None,START=None,COUNT=None,C=None,R:tl.constexpr=0,KEEP:tl.constexpr=False,QUANT:tl.constexpr=False):
    h=tl.program_id(2);qi=tl.program_id(0)*QM+tl.arange(0,QM)
    pg=tl.program_id(1)*GP+tl.arange(0,GP)
    proto=tl.arange(0,GP*16);pages=tl.program_id(1)*GP+proto//16
    d=tl.arange(0,64);p=tl.arange(0,16)
    q=tl.load(Q+(h*N+qi[:,None])*64+d[None,:],qi[:,None]<N,0.)
    key=tl.load(K+((h*B+pages[None,:])*16+proto[None,:]%16)*64+d[:,None],pages[None,:]<B,0.)
    raw=tl.dot(q,key).reshape((QM,GP,16))
    lc=tl.load(LC+(h*B+pg[:,None])*16+p[None,:],pg[:,None]<B,float('-inf'))
    logits=raw/math.sqrt(128)+lc[None,:,:]
    maximum=tl.max(logits,2);maximum=tl.where(pg[None,:]<B,maximum,0.)
    w=tl.exp(logits-maximum[:,:,None]);den=tl.sum(w,2)
    den=tl.where(pg[None,:]<B,den,1.)
    score=maximum+tl.log(den);w=w/den[:,:,None]
    wt=tl.permute(w,(1,0,2))
    if CONTENT_ENERGY:
        vd=tl.arange(0,32)
        value=tl.load(V+((h*B+pg[:,None,None])*16+p[None,:,None])*32+vd[None,None,:],pg[:,None,None]<B,0.)
        content=tl.dot(wt.to(tl.float16),value)
        energy=tl.trans(tl.sum(content*content,2))
        stored=content
        if QUANT:
            scale=tl.load(VS+h*B+pg,pg<B,1.)
            stored=nearbyint(content/scale[:,None,None]).to(tl.int8)
        ptr=((h*N+qi[None,:,None])*B+pg[:,None,None])*32+vd[None,None,:]
        tl.store(W+ptr,stored,(qi[None,:,None]<N)&(pg[:,None,None]<B))
    else:
        gram=tl.load(G+(h*B+pg[:,None,None])*256+p[None,:,None]*16+p[None,None,:],pg[:,None,None]<B,0.)
        wg=tl.dot(wt.to(tl.float16),gram.to(tl.float16))
        energy=tl.trans(tl.sum(wg*wt,2))
    tl.store(S+(h*N+qi[:,None])*B+pg[None,:],score,(qi[:,None]<N)&(pg[None,:]<B))
    tl.store(E+(h*N+qi[:,None])*B+pg[None,:],energy,(qi[:,None]<N)&(pg[None,:]<B))


