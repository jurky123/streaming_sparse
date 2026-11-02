"""Exact vectorized normalization on the verified summary-reuse runtime."""
from .gap_summary_reuse_runtime import GapSummaryReuseRuntime
from .gap_vector_stats import stats
from . import gap_feature_runtime as module
class GapVectorRuntime(GapSummaryReuseRuntime):
    def decision(self,*args,**kwargs):
        old=module.stats
        try:
            module.stats=stats
            return super().decision(*args,**kwargs)
        finally:module.stats=old
    def finish(self):return super().finish()|dict(stats_implementation='vectorized four-representative normalizers, native arithmetic')
