from .gap_tail_bf16qk_stage2_runtime import GapTailBF16QKStage2Runtime
from .gap_tensor_gram_runtime import GapTensorGramRuntime
from .gap_route_pick_ops import postprocess
from .gap_route_bf16qk import project
from . import gap_route_tensor_runtime as module
class GapRouteCombinedRuntime(GapTailBF16QKStage2Runtime):
    def decision(self,*args,**kwargs):
        old_post,old_project=module.postprocess,module.project
        try:
            module.postprocess=postprocess;module.project=project
            return GapTensorGramRuntime.decision(self,*args,**kwargs)
        finally:module.postprocess=old_post;module.project=old_project
    def finish(self):return super().finish()|dict(initial_pick='finite-sentinel fused selection',route_projection='BF16 high/low Tensor Core projection')
