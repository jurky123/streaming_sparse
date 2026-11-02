"""Experimental algebraically equivalent compensation; requires quality acceptance."""
import torch
from .gap_vector_runtime import GapVectorRuntime
from .gap_flat_tail import compensate
from . import gap_feature_runtime as module

def flat_native(q,k,v,pk,base,detail,basis,center,scale,lut,kernel):
    _,n,h,d=q.shape;out=torch.empty_like(q);lse=torch.empty(1,h,n,device=q.device)
    kernel[(n//64,h)](q,k,v,d**-.5,lut.shape[-1],lut,lse,out,h,n,k.shape[1],n//64,d,64,64,num_warps=4,num_stages=3)
    compensate(q,pk,base,detail,lut,lut,out,lse,basis,center,torch.ones(h,device=q.device),scale)
    return out
class GapFlatRuntime(GapVectorRuntime):
    def forward(self,*args,**kwargs):
        if self.training_event:return super().forward(*args,**kwargs)
        old=module.native
        try:
            module.native=flat_native
            return super().forward(*args,**kwargs)
        finally:module.native=old
    def finish(self):return super().finish()|dict(compensation_implementation='page-local decoded representatives, flattened online softmax; changed FP summation')
