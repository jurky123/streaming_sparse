"""Expose query tile size and precision using existing generalized shared kernel."""
import torch,triton as tr
from ..opt23.attention import _shared
from ..opt21.attention import _merge_restore
from ..opt16.grouping import restore

def attention(q,kv,ids,quant,seg,ix,group=32,splits=2,warps=2,stages=2,int8=True):
 h,n,d=q.shape;b=kv.state['key'].shape[1]
 out=torch.empty_like(q)
 if splits==1:
  _shared[(tr.cdiv(n,group),h,1)](q,quant.k if int8 else kv.k,kv.v,quant.scale,ids,out,out,n,kv.k.shape[1],b,group,ids.shape[1],ids.shape[2],1,int8,seg,num_warps=warps,num_stages=stages)
  return restore(out,ix)
 partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32);lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
 _shared[(tr.cdiv(n,group),h,splits)](q,quant.k if int8 else kv.k,kv.v,quant.scale,ids,partial,lse,n,kv.k.shape[1],b,group,ids.shape[1],ids.shape[2],splits,int8,seg,num_warps=warps,num_stages=stages)
 _merge_restore[(tr.cdiv(n,8),h)](partial,lse,ix,out,n,splits,8,True,num_warps=4,enable_fp_fusion=False)
 return out
