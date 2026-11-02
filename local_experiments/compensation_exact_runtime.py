"""Prepacked TF32 pairs, original 16-query/16-page accumulation ordering."""
from .compensation_optimized_runtime import OptimizedCompensationRuntime
from .compensation_tf32pack_kernels import compensate
class ExactCompensationRuntime(OptimizedCompensationRuntime):
    def __init__(self,engine,backend,config,audit=False):
        assert backend=='tf32pack' and tuple(config)==(16,16,4,1,False)
        super().__init__(engine,'fast',config,audit)
        self.optimized=compensate;self.method='rank8_tf32pack'
