import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import erf
@tr.jit
def _finish(T,ST,S,R,A,W1,W2,B2,OUT,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,G:tl.constexpr,APPROX:tl.constexpr):
    head=tl.program_id(1);q=tl.program_id(0)*G+tl.arange(0,G);row=head*N+q;a=tl.arange(0,128);d=tl.arange(0,16);p=tl.arange(0,P)
    si=tl.where(a<17,a-9,(a-17)//8);sj=tl.where(a<9,a-1,(a-17)%8)
    hasi=(a>=9)&(a<81);hasj=((a>=1)&(a<=8))|((a>=17)&(a<81))
    i=tl.load(R+row[:,None]*8+si[None,:],(q[:,None]<N)&hasi[None,:],-1);j=tl.load(A+row[:,None]*8+sj[None,:],(q[:,None]<N)&hasj[None,:],-1)
    si=tl.where(hasi,si,8);sj=tl.where(hasj,sj,8)
    base=tl.load(T+row[:,None]*384+d[None,:],q[:,None]<N,0.)
    yi=tl.load(T+row[:,None,None]*384+16+si[None,:,None]*16+d[None,None,:],q[:,None,None]<N,0.)
    yj=tl.load(T+row[:,None,None]*384+160+sj[None,:,None]*16+d[None,None,:],q[:,None,None]<N,0.)
    cross=tl.load(T+row[:,None]*384+304+(a[None,:]-17),(q[:,None]<N)&(a[None,:]>=17)&(a[None,:]<81),0.)
    zero=tl.load(T+row*384+368,q<N,0.);cross=tl.where((a>=17)&(a<81),cross,zero[:,None])
    w5=tl.load(W1+head*384+d*24+5).to(tl.float16).to(tl.float32)
    y=base[:,None,:]+yi+yj+cross[:,:,None]*w5[None,None,:]
    if APPROX:
        z=.7978845608028654*(y+.044715*y*y*y)
        act=tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;",constraints="=f,f",args=[z],dtype=tl.float32,is_pure=True,pack=1)
        y=.5*y*(1.+act)
    else:y=.5*y*(1.+erf(y*.7071067811865476))
    pred=tl.sum(y*tl.load(W2+head*16+d)[None,None,:],2)+tl.load(B2+head)
    count=tl.load(ST+row*80+71,q<N,0).to(tl.int32)
    valid=(a==0)|((a>=1)&(a<=8)&(j>=0))|((a>=9)&(a<=16)&(i>=0)&(count[:,None]>1))|((a>=17)&(a<81)&(i>=0)&(j>=0))
    pred=tl.where(a==0,0.,tl.where(valid,pred,float('-inf')))
    best=tl.max(pred,1);chosen=tl.min(tl.where(pred==best[:,None],a[None,:],2147483647),1)
    ri=tl.sum(tl.where(a[None,:]==chosen[:,None],i,0),1);aj=tl.sum(tl.where(a[None,:]==chosen[:,None],j,0),1)
    selected=tl.load(S+row[:,None]*B+p[None,:],(q[:,None]<N)&(p[None,:]<B),False)
    tl.store(OUT+row[:,None]*B+p[None,:],(selected&(p[None,:]!=ri[:,None]))|(p[None,:]==aj[:,None]),(q[:,None]<N)&(p[None,:]<B))
