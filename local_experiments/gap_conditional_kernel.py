"""Fused Haar projection and conditional V decoding, experimental native-order audit."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _conditional(W,BASE,DETAIL,O,C,P:tl.constexpr,T:tl.constexpr,N:tl.constexpr,B:tl.constexpr,PAIR:tl.constexpr,DEBUG:tl.constexpr,NATIVE:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);h=i//(T*P);d=tl.arange(0,8)
    if NATIVE:
        p=i//T%P;t=i%T;wi=(h*T+t)*P+p
    else:
        p=i%P;wi=i
    a=tl.load(W+wi*4,i<N,0);b=tl.load(W+wi*4+1,i<N,0);c=tl.load(W+wi*4+2,i<N,0);e=tl.load(W+wi*4+3,i<N,0)
    if PAIR:
        c0=(a+b)-(c+e);c1=(a-b)+(c-e);c2=(a-b)-(c-e)
    else:
        c0=((a+b)-c)-e;c1=((a-b)+c)-e;c2=((a-b)-c)+e
    if DEBUG:
        tl.store(C+wi*3,c0,i<N);tl.store(C+wi*3+1,c1,i<N);tl.store(C+wi*3+2,c2,i<N)
    off=(h[:,None]*P+p[:,None])*24+d[None,:]
    v0=tl.load(DETAIL+off,i[:,None]<N,0).to(tl.float32);v1=tl.load(DETAIL+off+8,i[:,None]<N,0).to(tl.float32);v2=tl.load(DETAIL+off+16,i[:,None]<N,0).to(tl.float32)
    v=c0[:,None]*v0;v=tl.fma(c1[:,None],v1,v);v=tl.fma(c2[:,None],v2,v)
    base=tl.load(BASE+(h[:,None]*P+p[:,None])*8+d[None,:],i[:,None]<N,0).to(tl.float32)
    tl.store(O+i[:,None]*8+d[None,:],base+v,i[:,None]<N)
def conditional(weight,base,detail,block=128,pair=False,debug=False,native_layout=True):
    h,t,p,_=weight.shape;out=torch.empty(h,p,t,8,device=weight.device).transpose(1,2) if native_layout else torch.empty(h,t,p,8,device=weight.device);coeff=torch.empty(h,t,p,3,device=weight.device) if debug else out
    _conditional[(tr.cdiv(h*t*p,block),)](weight,base,detail,out,coeff,p,t,h*t*p,block,pair,debug,native_layout,num_warps=4,enable_fp_fusion=False)
    return (out,coeff) if debug else out
