import os
from functools import partial
from .gap_direct_summary_runtime import GapDirectSummaryRuntime
from .gap_precision_tail import compensate
from . import gap_flat_runtime as module
class GapPrecisionRuntime(GapDirectSummaryRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tail_mode=int(os.environ.get('GAP_TAIL_MODE','3'))
        if self.tail_mode not in (1,3):raise ValueError(self.tail_mode)
    def forward(self,*args,**kwargs):
        if self.training_event:return super().forward(*args,**kwargs)
        old=module.compensate
        try:
            module.compensate=partial(compensate,mode=self.tail_mode)
            return super().forward(*args,**kwargs)
        finally:module.compensate=old
    def finish(self):
        return super().finish()|dict(tail_precision_mode=self.tail_mode,recovery_training=False)
