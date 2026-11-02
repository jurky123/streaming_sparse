"""Single BF16 representative QK product; TF32 AV retained, approximate."""
from functools import partial
from .gap_pool_linear_runtime import GapPoolLinearRuntime
from .gap_direct_summary_runtime import GapDirectSummaryRuntime
from .gap_tail_single import compensate
from . import gap_flat_runtime as module
class GapTailSingleRuntime(GapPoolLinearRuntime):
    def forward(self,*args,**kwargs):
        if self.training_event:return super().forward(*args,**kwargs)
        old=module.compensate
        try:
            module.compensate=partial(compensate,mode=1)
            return GapDirectSummaryRuntime.forward(self,*args,**kwargs)
        finally:module.compensate=old
    def finish(self):return super().finish()|dict(tail_qk='single BF16 projection; omits representative low-part product; approximate')
