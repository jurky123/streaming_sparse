"""Fuse four-representative normalizers without changing the native arithmetic."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _scaled_stats(X,Z,W,N:tl.constexpr,SOFT_ORDER:tl.constexpr,DIV:tl.constexpr,EXP:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    r=tl.arange(0,4)
    x=tl.load(X+i[:,None]*4+r[None,:],i[:,None]<N,0)
    x=x*0.08838834764831843
    even,odd=tl.split(tl.reshape(x,(B,2,2)))
    a,c=tl.split(even);b,d=tl.split(odd)
    mx=tl.maximum(tl.maximum(a,b),tl.maximum(c,d))
    ea=tl.exp(a-mx);eb=tl.exp(b-mx);ec=tl.exp(c-mx);ed=tl.exp(d-mx)
    tl.store(Z+i,(tl.log((ea+eb)+(ec+ed))+mx)+2.772588722239781,i<N)
    if EXP==1:ea=tl.exp(a-mx);eb=tl.exp(b-mx);ec=tl.exp(c-mx);ed=tl.exp(d-mx)
    if SOFT_ORDER==0:s=(ea+eb)+(ec+ed)
    elif SOFT_ORDER==1:s=(ea+ec)+(eb+ed)
    else:s=((ea+eb)+ec)+ed
    if DIV==0:
        inv=tl.div_rn(1.,s);ea=ea*inv;eb=eb*inv;ec=ec*inv;ed=ed*inv
    elif DIV==1:inv=1./s;ea=ea*inv;eb=eb*inv;ec=ec*inv;ed=ed*inv
    else:ea=ea/s;eb=eb/s;ec=ec/s;ed=ed/s
    values=tl.reshape(tl.join(tl.join(ea,ec),tl.join(eb,ed)),(B,4))
    tl.store(W+i[:,None]*4+r[None,:],values,i[:,None]<N)
def scaled_stats(logits,order=1,div=1,exp=0,block=256,warps=4):
    z=torch.empty(logits.shape[:-1],device=logits.device);w=torch.empty_like(logits)
    _scaled_stats[(tr.cdiv(z.numel(),block),)](logits,z,w,z.numel(),order,div,exp,block,num_warps=warps,enable_fp_fusion=False)
    return z,w
