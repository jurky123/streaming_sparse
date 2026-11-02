"""Original pairwise representative sum with BF16-sided TF32x2 products."""
from .compensation_optimized_runtime import OptimizedCompensationRuntime
from .compensation_stable_kernels import compensate
class StableCompensationRuntime(OptimizedCompensationRuntime):
    def __init__(self,engine,backend,config,audit=False):
        assert backend=='stable' and tuple(config)==(64,16,4,1,False)
        super().__init__(engine,'fast',config,audit)
        self.optimized=compensate;self.method='rank8_stable'
