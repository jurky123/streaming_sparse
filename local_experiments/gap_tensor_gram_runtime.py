from .gap_group_fused_runtime import GapGroupFusedRuntime
from .gap_tensor_gram import balanced_tensor_gram
from .gap_commit_graph_runtime import GapCommitGraphRuntime
from . import gap_commit_graph_runtime as module
class GapTensorGramRuntime(GapGroupFusedRuntime):
    def commit(self,*args,**kwargs):
        old=module.balanced
        try:
            module.balanced=balanced_tensor_gram
            return GapCommitGraphRuntime.commit(self,*args,**kwargs)
        finally:module.balanced=old
    def finish(self):
        return super().finish()|dict(grouping='split-K TF32x3 Gram for >512 tokens; fused direct iterations for <=512; stable sort')
