from .gap_direct_summary_runtime import GapDirectSummaryRuntime
from .gap_simt8_tail import compensate
from . import gap_flat_runtime as module
class GapSimtRuntime(GapDirectSummaryRuntime):
    def forward(self,*args,**kwargs):
        if self.training_event:return super().forward(*args,**kwargs)
        old=module.compensate
        try:
            module.compensate=compensate
            return super().forward(*args,**kwargs)
        finally:module.compensate=old
    def finish(self):return super().finish()|dict(tail_av='SIMT FP32 multiply/reduce, changed summation order, requires quality screen')
