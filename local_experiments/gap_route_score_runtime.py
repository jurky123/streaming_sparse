from .gap_tensor_gram_runtime import GapTensorGramRuntime
from .gap_route_score_scalar_ops import postprocess
from . import gap_route_tensor_runtime as module
class GapRouteScoreRuntime(GapTensorGramRuntime):
    def decision(self,*args,**kwargs):
        old=module.postprocess
        try:
            module.postprocess=postprocess
            return super().decision(*args,**kwargs)
        finally:module.postprocess=old
    def finish(self):
        return super().finish()|dict(route_score='fused features and scalar weighted reduction; changed summation order')
