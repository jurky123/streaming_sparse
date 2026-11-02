"""Native-order vector pooling for current summaries and history preparation."""
from .gap_route_half_runtime import GapRouteHalfRuntime
from .gap_pool_linear import weighted
from . import gap_feature_pool_ops as current_ops
from . import gap_direct_summary_ops as history_ops

def vector_weighted(k,v,a,block=32):return weighted(k,v,a,block=128)
class GapPoolLinearRuntime(GapRouteHalfRuntime):
    def current_summary(self,*args,**kwargs):
        if self.training_event:return super().current_summary(*args,**kwargs)
        old=current_ops.weighted
        try:
            current_ops.weighted=vector_weighted
            return super().current_summary(*args,**kwargs)
        finally:current_ops.weighted=old
    def commit(self,*args,**kwargs):
        if self.training_event:return super().commit(*args,**kwargs)
        old=history_ops.weighted
        try:
            history_ops.weighted=vector_weighted
            return super().commit(*args,**kwargs)
        finally:history_ops.weighted=old
    def finish(self):return super().finish()|dict(weighted_pool='contiguous128 native-order vector accumulators, current and history')
