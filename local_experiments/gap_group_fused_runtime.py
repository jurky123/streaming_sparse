"""Small-group fused power iterations on the latest summary inference candidate."""
from functools import partial
from .gap_feature_pool_runtime import GapFeaturePoolRuntime
from .gap_group_fused import balanced_fused
from .gap_commit_graph_runtime import GapCommitGraphRuntime
from . import gap_commit_graph_runtime as module
class GapGroupFusedRuntime(GapFeaturePoolRuntime):
    def commit(self,*args,**kwargs):
        old=module.balanced
        try:
            module.balanced=partial(balanced_fused,threshold=512)
            return GapCommitGraphRuntime.commit(self,*args,**kwargs)
        finally:module.balanced=old
    def finish(self):
        return super().finish()|dict(grouping='fused power iterations for groups <=512, stable native sort, hard page-local representatives')
