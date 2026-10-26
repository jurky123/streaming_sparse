"""Opt9 Compact attention consuming a precomputed voted union."""
import torch
import triton as tr
from ..opt9.segmented_attention import _quantize,_split,_merge

def attention(q,kh,vh,kc,vc,mask,group=32,splits=2,warps=2,stages=2,return_union=False,pack_v=False,fuse_pack=False):
    h,n,d=q.shape;b=mask.shape[-1];nh=kh.shape[1];nk=nh+kc.shape[1]
    original_vh=vh
    if pack_v:
        vh=torch.empty((h,nk,d),device=q.device,dtype=vh.dtype)if fuse_pack else torch.cat((vh,vc),1)
    ki=torch.empty((h,nk,d),device=q.device,dtype=torch.int8)
    ks=torch.empty((h,b),device=q.device,dtype=torch.float32)
    _quantize[(b,h)](kh,kc,ki,ks,original_vh,vc,vh,nk,b,kh.stride(0),kh.stride(1),kc.stride(0),kc.stride(1),nh,original_vh.stride(0),original_vh.stride(1),vc.stride(0),vc.stride(1),pack_v and fuse_pack,num_warps=4)
    assert group==32
    ids,counts=mask._opt13_union
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
    lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
    _split[(tr.cdiv(n,group),h,splits)](q,ki,vh,vc,ks,mask,ids,counts,partial,lse,n,nk,b,
        q.stride(0),q.stride(1),ki.stride(0),ki.stride(1),vh.stride(0),vh.stride(1),vc.stride(0),vc.stride(1),nh,group,splits,False,pack_v,
        num_warps=warps,num_stages=stages)
    _merge[(h*n,)](partial,lse,out,n,splits,num_warps=4)
    return (out,counts)if return_union else out
