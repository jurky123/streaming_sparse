from .gap_tail_bf16qk_stage2_runtime import GapTailBF16QKStage2Runtime
from .gap_tensor_gram_runtime import GapTensorGramRuntime
from .gap_route_pick_ops import postprocess
from . import gap_route_tensor_runtime as module
class GapRoutePickRuntime(GapTailBF16QKStage2Runtime):
    def decision(self,*args,**kwargs):
        old=module.postprocess
        try:
            module.postprocess=postprocess
            return GapTensorGramRuntime.decision(self,*args,**kwargs)
        finally:module.postprocess=old
    def finish(self):return super().finish()|dict(initial_pick='fused top-k/mask/IDs; finite sentinels; low-index ties')
