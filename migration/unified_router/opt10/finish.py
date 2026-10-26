import triton as tr
import triton.language as tl
from triton.language.extra.cuda.libdevice import erf
@tr.jit
def _finish(T,ST,S,R,A,W1,W2,B2,OUT,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,APPROX:tl.constexpr):
    row=tl.program_id(0);head=row//N;a=tl.arange(0,128);d=tl.arange(0,16);p=tl.arange(0,P)
    si=tl.where(a<17,a-9,(a-17)//8);sj=tl.where(a<9,a-1,(a-17)%8)
    hasi=(a>=9)&(a<81);hasj=((a>=1)&(a<=8))|((a>=17)&(a<81))
    i=tl.load(R+row*8+si,hasi,-1);j=tl.load(A+row*8+sj,hasj,-1)
    si=tl.where(hasi,si,8);sj=tl.where(hasj,sj,8)
    base=tl.load(T+row*384+d).to(tl.float32)
    yi=tl.load(T+row*384+16+si[:,None]*16+d[None,:]).to(tl.float32)
    yj=tl.load(T+row*384+160+sj[:,None]*16+d[None,:]).to(tl.float32)
    cross=tl.load(T+row*384+304+(a-17),(a>=17)&(a<81),0.).to(tl.float32)
    zero=tl.load(T+row*384+368).to(tl.float32);cross=tl.where((a>=17)&(a<81),cross,zero)
    w5=tl.load(W1+head*384+d*24+5).to(tl.float16).to(tl.float32)
    y=base[None,:]+yi+yj+cross[:,None]*w5[None,:]
    if APPROX:
        z=.7978845608028654*(y+.044715*y*y*y)
        act=tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;",constraints="=f,f",args=[z],dtype=tl.float32,is_pure=True,pack=1)
        y=.5*y*(1.+act)
    else:y=.5*y*(1.+erf(y*.7071067811865476))
    pred=tl.sum(y*tl.load(W2+head*16+d)[None,:],1)+tl.load(B2+head)
    count=tl.load(ST+row*80+71).to(tl.int32)
    valid=(a==0)|((a>=1)&(a<=8)&(j>=0))|((a>=9)&(a<=16)&(i>=0)&(count>1))|((a>=17)&(a<81)&(i>=0)&(j>=0))
    pred=tl.where(a==0,0.,tl.where(valid,pred,float('-inf')))
    best=tl.max(pred,0);chosen=tl.min(tl.where(pred==best,a,2147483647),0)
    ri=tl.sum(tl.where(a==chosen,i,0),0);aj=tl.sum(tl.where(a==chosen,j,0),0)
    selected=tl.load(S+row*B+p,p<B,False)
    tl.store(OUT+row*B+p,(selected&(p!=ri))|(p==aj),p<B)
