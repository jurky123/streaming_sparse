from .gap_route_score_runtime import GapRouteScoreRuntime
from .gap_route_contiguous_ops import postprocess
from . import gap_route_tensor_runtime as module
class GapRouteReduceRuntime(GapRouteScoreRuntime):
    def decision(self,*args,**kwargs):
        old=module.postprocess
        try:
            module.postprocess=postprocess
            # Bypass the score-only wrapper, which would replace this function.
            from .gap_tensor_gram_runtime import GapTensorGramRuntime
            return GapTensorGramRuntime.decision(self,*args,**kwargs)
        finally:module.postprocess=old
    def finish(self):
        return super().finish()|dict(route_moments='fused selected/all conditional-V reductions; contiguous conditional V; changed reduction order')
