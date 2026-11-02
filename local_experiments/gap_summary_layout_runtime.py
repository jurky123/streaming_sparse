import types
from .gap_input_runtime import GapInputRuntime
from .gap_summary_layout_ops import layout_pool
class GapSummaryLayoutRuntime(GapInputRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for m in self.model.layers.values():m.pool=types.MethodType(layout_pool,m)
    def finish(self):return super().finish()|dict(summary_group_layout='direct H,R,P,T,D centered outputs; removes two full tensor clones')
