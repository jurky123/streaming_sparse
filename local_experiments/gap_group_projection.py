"""Raw BF16-origin K/V projection followed by projected-space centering."""
import math,torch,triton as tr,triton.language as tl
from .gap_spatial_runtime import group_features as reference_features
from .gap_spatial_softmax import spatial
@tr.jit
def _project(X,W,O,M:tl.constexpr,XH:tl.constexpr,XT:tl.constexpr,WH:tl.constexpr,WN:tl.constexpr,WD:tl.constexpr,BM:tl.constexpr):
    h=tl.program_id(1);t=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.arange(0,128);n=tl.arange(0,32)
    x=tl.load(X+h*XH+t[:,None]*XT+d[None,:],t[:,None]<M,0).to(tl.bfloat16)
    w=tl.load(W+h*WH+n[None,:]*WN+d[:,None]*WD)
    hi=w.to(tl.bfloat16);lo=(w-hi.to(tl.float32)).to(tl.bfloat16)
    value=tl.dot(x,lo);value=tl.dot(x,hi,value)
    tl.store(O+(h*M+t[:,None])*32+n[None,:],value,t[:,None]<M)
def project(x,w):
    h,p,t,d=x.shape;assert d==128 and w.shape==(h,32,128)
    flat=x.reshape(h,p*t,d);assert flat.stride(-1)==1
    out=torch.empty(h,p,t,32,device=x.device)
    _project[(tr.cdiv(p*t,64),h)](flat,w,out,p*t,*flat.stride()[:2],*w.stride(),64,num_warps=4,num_stages=2)
    return out
def group_features(self,k,v):
    if torch.is_grad_enabled():return reference_features(self,k,v)
    # Commit receives raw BF16 K/V, possibly converted to FP32/reordered only.
    raw=project(k,self.bank);score=(raw-raw.mean(-2,keepdim=True))/math.sqrt(128)
    rms=score.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6);key=score/rms
    a=spatial(score,block=4 if k.shape[-2]>64 else 8)
    vp=project(v,self.group_v.transpose(1,2));mean=(a*vp).sum(-2,keepdim=True)
    response=a*k.shape[-2]*(vp-mean);response=response/response.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    return key+.1*self.group_gate.tanh()[:,None,None]*response
