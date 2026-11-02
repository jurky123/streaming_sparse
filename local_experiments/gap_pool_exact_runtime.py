"""Exact fused page-local K/V weighted reduction, including current/history summaries."""
import types
from .gap_ordered_runtime import GapFlatOrderedRuntime
from .gap_pool_ops import fused_pool
class GapPoolExactRuntime(GapFlatOrderedRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for m in self.model.layers.values():m.pool=types.MethodType(fused_pool,m)
    def finish(self):return super().finish()|dict(weighted_pool='native-order fused K/V weighted sum16, no product temporary')
