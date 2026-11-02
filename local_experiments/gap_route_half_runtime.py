"""FP16 conditional probabilities as transient route workspace; logits stay FP32."""
import torch
from .gap_route_fused_tuned_runtime import GapRouteFusedTunedRuntime
class GapRouteHalfRuntime(GapRouteFusedTunedRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # graw now stores only normalized within-page probabilities. The fused
        # kernel keeps accumulation/logz FP32; no FP16 logits are stored.
        self.graw=torch.empty_like(self.graw,dtype=torch.float16)
    def finish(self):return super().finish()|dict(route_within_dtype='float16 temporary probability workspace; FP32 logz')
