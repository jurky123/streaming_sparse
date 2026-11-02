"""Representative variance without materializing centered squared K."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
@tr.jit
def _kvar(K,O,D:tl.constexpr):
    page=tl.program_id(0);d=tl.arange(0,D)
    a=tl.load(K+page*4*D+d);b=tl.load(K+page*4*D+D+d);c=tl.load(K+page*4*D+2*D+d);e=tl.load(K+page*4*D+3*D+d)
    mean=(((a+b)+c)+e)*.25
    a=a-mean;b=b-mean;c=c-mean;e=e-mean
    variance=(((tl.sum(a*a,0)+tl.sum(b*b,0))+tl.sum(c*c,0))+tl.sum(e*e,0))/(4*D)
    tl.store(O+page,lib.log1p(variance))
def kvar(k):
    h,p,r,d=k.shape;assert r==4 and d==128 and k.is_contiguous()
    out=torch.empty(h,p,device=k.device)
    _kvar[(h*p,)](k,out,d,num_warps=4,enable_fp_fusion=False)
    return out
