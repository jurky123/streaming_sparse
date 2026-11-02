"""Raw-input Tensor Core feature projection only inside history preparation."""
import types
from .gap_pool_linear_runtime import GapPoolLinearRuntime
from .gap_group_projection import group_features
class GapGroupProjectionRuntime(GapPoolLinearRuntime):
    def commit(self,l,*args,**kwargs):
        if self.training_event:return super().commit(l,*args,**kwargs)
        m=self.model[l];old=m.group_features
        try:
            m.group_features=types.MethodType(group_features,m)
            return super().commit(l,*args,**kwargs)
        finally:m.group_features=old
    def finish(self):return super().finish()|dict(history_features='raw BF16-origin projection, learned FP32 weights split high/low, projected-space centering; approximate')
