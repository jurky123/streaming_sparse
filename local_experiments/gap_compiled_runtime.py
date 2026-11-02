"""Experimental fused route; cannot deploy unless actual rollout audit passes."""
import torch
from . import gap_graph_runtime as graph_module
from .gap_fast_runtime import GapFastRuntime
_reference=graph_module.postprocess
_compiled=torch.compile(_reference,fullgraph=True,options={'triton.cudagraphs':False})
class GapCompiledRuntime(GapFastRuntime):
    def __init__(self,*args,**kwargs):
        graph_module.postprocess=_compiled
        super().__init__(*args,**kwargs)
