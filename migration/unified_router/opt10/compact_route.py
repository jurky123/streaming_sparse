from .compact_weights import CompactWeights
from .tuned_route import TunedMomentRoute
from .fused_route import FusedMomentRoute
class CompactMomentRoute(CompactWeights,TunedMomentRoute):pass
class CompactFusedRoute(CompactWeights,FusedMomentRoute):pass
