"""Repository page LUT exact attention + full-mean/rank8 batched compensation."""
import math
import torch
from ..gap_exact_round_kernel import _attn_fwd
from ..review_v1.ops import haar

class Workspace:
    def __init__(self):self.entries={}
    def tensor(self,name,shape,device,dtype):
        key=(name,tuple(shape),device,dtype)
        if key not in self.entries:self.entries[key]=torch.empty(shape,device=device,dtype=dtype)
        return self.entries[key]
    def bytes(self):return sum(x.numel()*x.element_size() for x in self.entries.values())

@torch.no_grad()
def exact(q,k,v,ids,workspace):
    """Native B=1,L,H,D contiguous KV; never gather/repack the history."""
    _,n,h,d=q.shape;p=k.shape[1]//64;count=ids.shape[-1]
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    out=workspace.tensor('exact',(1,n,h,d),q.device,torch.float32)
    lse=workspace.tensor('exact_lse',(h,n),q.device,torch.float32)
    lut=ids.contiguous()[None]
    _attn_fwd[(n//64,h)](q,k,v,d**-.5,count,lut,lse,out,h,n,p*64,n//64,d,64,64,HIGH_PRECISION_AV=True,num_warps=4,num_stages=2)
    return out,lse*math.log(2)  # Repository kernel returns base-2 LSE.

@torch.no_grad()
def compensate(logits,summary,basis,mask,haar_basis=None,matmul=torch.bmm):
    # logits H,T,Q,P,R; no Q,P,128 Value expansion.
    h,t,n,p,r=logits.shape
    z=logits.masked_fill(mask[:,:,None,:,None],-torch.inf)
    lse=z.flatten(-2).logsumexp(-1)
    finite=torch.isfinite(lse);safe=torch.where(finite,lse,torch.zeros_like(lse))
    w=(z-safe[...,None,None]).exp()
    mean=matmul(w.sum(-1).reshape(h,t*n,p),summary.mean.float())
    contrasts=torch.einsum('htqpr,ar->htqpa',w,haar(logits) if haar_basis is None else haar_basis)
    code=matmul(contrasts.reshape(h,t*n,p*3),summary.detail.float().flatten(1,2))
    out=mean+torch.bmm(code,basis.bfloat16().float().transpose(1,2))
    return out,lse.reshape(h,t*n)+math.log(16)

@torch.no_grad()
def merge(exact_output,exact_lse,approx,approx_lse):
    le=exact_lse;la=approx_lse;total=torch.logaddexp(le,la)
    oe=exact_output[0].transpose(0,1)
    out=(le-total).exp()[...,None]*oe+(la-total).exp()[...,None]*approx
    return out.transpose(0,1)[None].to(torch.bfloat16)


def compensate_tensor(logits,summary,basis,mask,haar_basis=None):
    from .tensor_bmm import tensor_bmm
    return compensate(logits,summary,basis,mask,haar_basis,matmul=tensor_bmm)
