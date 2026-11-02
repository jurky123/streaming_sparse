"""Retile fused projection/stats, preserving BF16 dot and normalizer arithmetic."""
import triton as tr
from .gap_route_bf16_stats import _project_stats
from .gap_route_fused_stats_runtime import GapRouteFusedStatsRuntime
from . import gap_route_fused_stats_runtime as module

def project_stats(q,k,z,w):
    h,m,d=q.shape;n=k.shape[1]
    assert q.is_contiguous() and k.stride(-1)==1 and z.is_contiguous() and w.is_contiguous() and n%4==0
    return _project_stats[(tr.cdiv(m,128),tr.cdiv(n,128),h)](q,k,z,w,m,n,d,*k.stride()[:2],128,128,128,2,num_warps=8,num_stages=2,enable_fp_fusion=False)
class GapRouteFusedTunedRuntime(GapRouteFusedStatsRuntime):
    def decision(self,*args,**kwargs):
        if self.training_event:return super().decision(*args,**kwargs)
        old=module.project_stats
        try:
            module.project_stats=project_stats
            return super().decision(*args,**kwargs)
        finally:module.project_stats=old
    def finish(self):return super().finish()|dict(route_fused_tile='128x128x128, 8warps, 2stages')
