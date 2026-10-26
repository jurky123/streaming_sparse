"""Schedule-only tuning of the existing per-query masked sparse attention."""
import torch
import triton as tr
from ..fast_kernels import _union,_attention

def attention(q,k,v,mask,group=16,warps=4,stages=2):
    h,n,d=q.shape;b=mask.shape[-1]
    ids=torch.empty((h,tr.cdiv(n,group),b),device=q.device,dtype=torch.int32)
    counts=torch.empty(ids.shape[:2],device=q.device,dtype=torch.int32)
    _union[(tr.cdiv(n,group),h)](mask,ids,counts,n,b,group,tr.next_power_of_2(b),num_warps=4)
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    _attention[(tr.cdiv(n,group),h)](q,k,v,mask,ids,counts,out,n,k.shape[1],b,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),group,
        num_warps=warps,num_stages=stages)
    return out
