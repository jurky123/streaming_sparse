"""Full-page inference specialization with original scale rounding preserved."""
from .gap_history_cov_runtime import GapHistoryCovRuntime
from .gap_exact_round_kernel import _attn_fwd
class GapExactRoundRuntime(GapHistoryCovRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.original_exact_kernel=self.bare_kernel
        self.bare_kernel=_attn_fwd
    def forward(self,*args,**kwargs):
        if not self.training_event:return super().forward(*args,**kwargs)
        old=self.bare_kernel
        try:
            self.bare_kernel=self.original_exact_kernel
            return super().forward(*args,**kwargs)
        finally:self.bare_kernel=old
    def finish(self):return super().finish()|dict(exact_kernel='full page64 int32; explicit mul.rn scale boundary')
