"""Extend fused covariance only to commit's BF16-origin reordered raw K/V."""
import types
from .gap_route_combined_runtime import GapRouteCombinedRuntime
from .gap_covariance_kernel import covariance

def commit_covariance(self,k,v):
    # GapCommitGraphRuntime: k is reorder(kin.float()); v is reorder(vin).
    # No arithmetic changes raw BF16 values before this call.
    return covariance(k,v,self.basis)
class GapHistoryCovRuntime(GapRouteCombinedRuntime):
    def commit(self,l,*args,**kwargs):
        if self.training_event:return super().commit(l,*args,**kwargs)
        m=self.model[l];old=m.covariance
        try:
            m.covariance=types.MethodType(commit_covariance,m)
            return super().commit(l,*args,**kwargs)
        finally:m.covariance=old
    def finish(self):return super().finish()|dict(history_covariance='fused BF16 projection on BF16-origin FP32 reorder input; changed accumulation order')
