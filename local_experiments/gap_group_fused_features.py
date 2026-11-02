"""Fused learned history response features; no retained extra summaries."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .gap_group_projection import project
from .gap_spatial_runtime import group_features as reference
@tr.jit
def _page(RAW,VP,GATE,O,P:tl.constexpr):
    row=tl.program_id(0);h=row//P;t=tl.arange(0,64);f=tl.arange(0,32)
    raw=tl.load(RAW+(row*64+t[:,None])*32+f[None,:]);score=(raw-tl.sum(raw,0)[None,:]/64.)*0.08838834764831843
    krms=tl.maximum(lib.sqrt(tl.sum(tl.sum(score*score,0),0)/2048.),1.e-6)
    e=tl.exp(score-tl.max(score,0)[None,:]);a=e/tl.sum(e,0)[None,:]
    vp=tl.load(VP+(row*64+t[:,None])*32+f[None,:]);mean=tl.sum(a*vp,0);response=a*64.*(vp-mean[None,:])
    rrms=tl.maximum(lib.sqrt(tl.sum(tl.sum(response*response,0),0)/2048.),1.e-6)
    gate=.1*lib.tanh(tl.load(GATE+h*32+f));value=score/krms+gate[None,:]*(response/rrms)
    tl.store(O+(row*64+t[:,None])*32+f[None,:],value)
@tr.jit
def _moments(RAW,VP,RESP,MEAN,KS,RS,T:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);f=tl.program_id(1)*B+tl.arange(0,B);t=tl.arange(0,R)
    off=(row*T+t[:,None])*32+f[None,:]
    raw=tl.load(RAW+off,t[:,None]<T,0);mean=tl.sum(raw,0)/T
    score=tl.where(t[:,None]<T,(raw-mean[None,:])*0.08838834764831843,0.)
    ks=tl.sum(score*score,0)
    x=tl.where(t[:,None]<T,score,-float('inf'));e=tl.exp(x-tl.max(x,0)[None,:]);a=e/tl.sum(e,0)[None,:]
    vp=tl.load(VP+off,t[:,None]<T,0);mu=tl.sum(a*vp,0);response=a*T*(vp-mu[None,:]);rs=tl.sum(response*response,0)
    tl.store(RESP+off,response,t[:,None]<T);tl.store(MEAN+row*32+f,mean);tl.store(KS+row*32+f,ks);tl.store(RS+row*32+f,rs)
@tr.jit
def _finish(RAW,RESP,MEAN,KS,RS,GATE,O,T:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);block=tl.program_id(1);i=block*B+tl.arange(0,B);f=i%32;features=tl.arange(0,32)
    krms=tl.maximum(lib.sqrt(tl.sum(tl.load(KS+row*32+features),0)/(T*32)),1.e-6)
    rrms=tl.maximum(lib.sqrt(tl.sum(tl.load(RS+row*32+features),0)/(T*32)),1.e-6)
    raw=tl.load(RAW+row*T*32+i,i<T*32,0);response=tl.load(RESP+row*T*32+i,i<T*32,0)
    mean=tl.load(MEAN+row*32+f);gate=.1*lib.tanh(tl.load(GATE+(row//P)*32+f))
    score=(raw-mean)*0.08838834764831843;value=score/krms+gate*(response/rrms)
    tl.store(O+row*T*32+i,value,i<T*32)
def group_features(self,k,v):
    if torch.is_grad_enabled() or k.shape[-2] not in [64,4608]:return reference(self,k,v)
    h,p,t,d=k.shape;raw=project(k,self.bank);vp=project(v,self.group_v.transpose(1,2));out=torch.empty_like(raw)
    if t==64:_page[(h*p,)](raw,vp,self.group_gate,out,p,num_warps=4,enable_fp_fusion=False)
    else:
        response=torch.empty_like(raw);mean=torch.empty(h,p,32,device=k.device);ks=torch.empty_like(mean);rs=torch.empty_like(mean)
        _moments[(h*p,16)](raw,vp,response,mean,ks,rs,t,tr.next_power_of_2(t),2,num_warps=8,enable_fp_fusion=False)
        _finish[(h*p,tr.cdiv(t*32,512))](raw,response,mean,ks,rs,self.group_gate,out,t,p,512,num_warps=4,enable_fp_fusion=False)
    return out
