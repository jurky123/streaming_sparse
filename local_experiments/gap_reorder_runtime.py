"""Exact gather specialization during history preparation; no new retained KV."""
import torch
from .gap_gain_vector64_runtime import GapGainVector64Runtime
from .gap_reorder_kernel import reorder
from . import gap_commit_graph_runtime as module

def specialized_reorder(x,order):
    block=32 if x.shape[-1]==32 else (8 if x.dtype==torch.float32 else 16)
    return reorder(x,order,block=block)

class GapReorderRuntime(GapGainVector64Runtime):
    def commit(self,*args,**kwargs):
        if self.training_event:return super().commit(*args,**kwargs)
        old=module.reorder
        try:
            module.reorder=specialized_reorder
            return super().commit(*args,**kwargs)
        finally:
            module.reorder=old
    def finish(self):
        return super().finish()|dict(history_reorder='Specialized row gather, same values and contiguous output; block32 feature/8 FP32 K/16 BF16 V')
