"""Fuse four-representative normalizers without changing the native arithmetic."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _stats(X,Z,W,N:tl.constexpr,SOFT_ORDER:tl.constexpr,DIV:tl.constexpr,EXP:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.load(X+i*4,i<N,0);b=tl.load(X+i*4+1,i<N,0);c=tl.load(X+i*4+2,i<N,0);d=tl.load(X+i*4+3,i<N,0)
    mx=tl.maximum(tl.maximum(a,b),tl.maximum(c,d))
    ea=lib.exp(a-mx);eb=lib.exp(b-mx);ec=lib.exp(c-mx);ed=lib.exp(d-mx)
    tl.store(Z+i,(lib.log((ea+eb)+(ec+ed))+mx)+2.772588722239781,i<N)
    if EXP==1:ea=tl.exp(a-mx);eb=tl.exp(b-mx);ec=tl.exp(c-mx);ed=tl.exp(d-mx)
    if SOFT_ORDER==0:s=(ea+eb)+(ec+ed)
    elif SOFT_ORDER==1:s=(ea+ec)+(eb+ed)
    else:s=((ea+eb)+ec)+ed
    if DIV==0:
        inv=tl.div_rn(1.,s);ea=ea*inv;eb=eb*inv;ec=ec*inv;ed=ed*inv
    elif DIV==1:ea=tl.div_rn(ea,s);eb=tl.div_rn(eb,s);ec=tl.div_rn(ec,s);ed=tl.div_rn(ed,s)
    else:ea=ea/s;eb=eb/s;ec=ec/s;ed=ed/s
    tl.store(W+i*4,ea,i<N);tl.store(W+i*4+1,eb,i<N);tl.store(W+i*4+2,ec,i<N);tl.store(W+i*4+3,ed,i<N)
def stats(logits,order=1,div=0,exp=0):
    z=torch.empty(logits.shape[:-1],device=logits.device);w=torch.empty_like(logits)
    _stats[(tr.cdiv(z.numel(),256),)](logits,z,w,z.numel(),order,div,exp,256,enable_fp_fusion=False)
    return z,w
