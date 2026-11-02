"""Avoid materializing all per-probe within-page probabilities."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _logz(X,Z,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);r=tl.arange(0,4)
    x=tl.load(X+i[:,None]*4+r[None,:],i[:,None]<N,0)
    even,odd=tl.split(tl.reshape(x,(B,2,2)));a,c=tl.split(even);b,d=tl.split(odd)
    mx=tl.maximum(tl.maximum(a,b),tl.maximum(c,d))
    ea=lib.exp(a-mx);eb=lib.exp(b-mx);ec=lib.exp(c-mx);ed=lib.exp(d-mx)
    tl.store(Z+i,(lib.log((ea+eb)+(ec+ed))+mx)+2.772588722239781,i<N)
@tr.jit
def _weight(PROB,X,MASS,OUT,P:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);r=tl.arange(0,4);g=tl.arange(0,4);ht=i//P;p=i%P
    acc=tl.full((B,4,4),0.,tl.float32)
    for j in tl.static_range(4):
        probe=g+4*j
        prob=tl.load(PROB+(ht[:,None]*16+probe[None,:])*P+p[:,None],i[:,None]<N,0)
        x=tl.load(X+((ht[:,None,None]*16+probe[None,:,None])*P+p[:,None,None])*4+r[None,None,:],i[:,None,None]<N,0)
        x=tl.reshape(x,(B*4,4))
        even,odd=tl.split(tl.reshape(x,(B*4,2,2)));a,c=tl.split(even);b,d=tl.split(odd)
        mx=tl.maximum(tl.maximum(a,b),tl.maximum(c,d))
        ea=lib.exp(a-mx);eb=lib.exp(b-mx);ec=lib.exp(c-mx);ed=lib.exp(d-mx)
        s=(ea+ec)+(eb+ed)
        ea=tl.div_rn(ea,s);eb=tl.div_rn(eb,s);ec=tl.div_rn(ec,s);ed=tl.div_rn(ed,s)
        w=tl.reshape(tl.join(tl.join(ea,ec),tl.join(eb,ed)),(B,4,4))
        acc=acc+prob[:,:,None]*w
    a=tl.sum(tl.where(g[None,:,None]==0,acc,0.),1);b=tl.sum(tl.where(g[None,:,None]==1,acc,0.),1)
    c=tl.sum(tl.where(g[None,:,None]==2,acc,0.),1);d=tl.sum(tl.where(g[None,:,None]==3,acc,0.),1)
    mass=tl.maximum(tl.load(MASS+i,i<N,1.),1.e-12)
    tl.store(OUT+i[:,None]*4+r[None,:],tl.div_rn((((a+b)+c)+d)/16.,mass[:,None]),i[:,None]<N)
def logz(x):
    out=torch.empty(x.shape[:-1],device=x.device)
    _logz[(tr.cdiv(out.numel(),256),)](x,out,out.numel(),256,num_warps=4,enable_fp_fusion=False)
    return out
def weight(prob,x,mass,block=64):
    h,t,m,p=prob.shape;assert m==16;out=torch.empty(h,t,p,4,device=prob.device)
    _weight[(tr.cdiv(h*t*p,block),)](prob,x,mass,out,p,h*t*p,block,num_warps=4,enable_fp_fusion=False)
    return out
