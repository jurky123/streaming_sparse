"""Inference summary candidates; unchanged persistent cache dimensions."""
import types,torch
from .gap_route_tensor_runtime import GapRouteTensorRuntime
from .gap_model import GapLayer
from .gap_covariance_kernel import covariance
from .gap_raw_pool_ops import raw_pool

def fast_covariance(self,k,v):
    # Only cached BF16 current pages use the candidate. Keep FP32 history inputs
    # and differentiable evaluation on the original implementation.
    if torch.is_grad_enabled() or k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16:
        return GapLayer.covariance(self,k,v)
    return covariance(k,v,self.basis)

class GapCovarianceRuntime(GapRouteTensorRuntime):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        for m in self.model.layers.values():m.covariance=types.MethodType(fast_covariance,m)
    def finish(self):
        return super().finish()|dict(covariance_implementation='BF16 current pages: fused Tensor Core projection and FP32 moments; FP32 history unchanged')

class GapSummaryLiteRuntime(GapCovarianceRuntime):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        for m in self.model.layers.values():m.pool=types.MethodType(raw_pool,m)
    def finish(self):
        return super().finish()|dict(current_pool='raw page-local K/V dot products; centered final logits; fused weights; changed FP arithmetic')
