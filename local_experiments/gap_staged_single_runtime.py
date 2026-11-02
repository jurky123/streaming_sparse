"""Compose measured full-route staging and single-BF16 compensation."""
from functools import partial
from .gap_route_staged_graph_runtime import GapRouteStagedGraphRuntime
from .gap_direct_summary_runtime import GapDirectSummaryRuntime
from .gap_tail_single import compensate
from . import gap_flat_runtime as module
class GapStagedSingleRuntime(GapRouteStagedGraphRuntime):
    def forward(self,*args,**kwargs):
        if self.training_event:return super().forward(*args,**kwargs)
        old=module.compensate
        try:
            module.compensate=partial(compensate,mode=1)
            return GapDirectSummaryRuntime.forward(self,*args,**kwargs)
        finally:module.compensate=old
    def finish(self):return super().finish()|dict(tail_qk='single BF16 representative projection; approximate',combined_route_tail=True)
