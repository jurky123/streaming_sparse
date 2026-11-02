import types
from .gap_summary_lite_runtime import GapSummaryLiteRuntime
from .gap_feature_pool_ops import feature_pool
class GapFeaturePoolRuntime(GapSummaryLiteRuntime):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        for m in self.model.layers.values():m.pool=types.MethodType(feature_pool,m)
    def finish(self):
        return super().finish()|dict(current_pool='fused projected centering, feature RMS and learned page-local weights; no full current feature intermediates')
