"""Transient representative packing; no additional persistent KV summaries."""
from functools import partial
from .gap_exact_round_runtime import GapExactRoundRuntime
from .gap_direct_summary_runtime import GapDirectSummaryRuntime
from .gap_tail_packed import compensate
from . import gap_flat_runtime as module
class GapTailPackedRuntime(GapExactRoundRuntime):
    def forward(self,*args,**kwargs):
        if self.training_event:return super().forward(*args,**kwargs)
        old=module.compensate
        try:
            module.compensate=partial(compensate,mode=1)
            return GapDirectSummaryRuntime.forward(self,*args,**kwargs)
        finally:module.compensate=old
    def finish(self):return super().finish()|dict(tail_packing='transient BF16 high/low preprocessing included per call')
