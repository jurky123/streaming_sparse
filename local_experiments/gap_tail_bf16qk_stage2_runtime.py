from functools import partial
from .gap_route_reduce_runtime import GapRouteReduceRuntime
from .gap_direct_summary_runtime import GapDirectSummaryRuntime
from .gap_tail_bf16qk_stage2 import compensate
from . import gap_flat_runtime as module
class GapTailBF16QKStage2Runtime(GapRouteReduceRuntime):
    def forward(self,*args,**kwargs):
        if self.training_event:return super().forward(*args,**kwargs)
        old=module.compensate
        try:
            module.compensate=partial(compensate,mode=1)
            # Bypass precision wrapper that would reinstall the TF32-QK kernel.
            return GapDirectSummaryRuntime.forward(self,*args,**kwargs)
        finally:module.compensate=old
    def finish(self):
        return super().finish()|dict(tail_qk='two BF16 high/low products, stage2; TF32 single-term AV')
