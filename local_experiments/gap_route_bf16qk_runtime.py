from .gap_tail_bf16qk_stage2_runtime import GapTailBF16QKStage2Runtime
from .gap_route_bf16qk import project
from . import gap_route_tensor_runtime as module
class GapRouteBF16QKRuntime(GapTailBF16QKStage2Runtime):
    def decision(self,*args,**kwargs):
        old=module.project
        try:
            module.project=project
            return super().decision(*args,**kwargs)
        finally:module.project=old
    def finish(self):return super().finish()|dict(route_projection='BF16 two-term K split; BF16 queries; changed FP logits')
