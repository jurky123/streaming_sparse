"""Opt-in Sparge32 runtime. Frozen weights and20% routing; numerical changes.

Call under torch.no_grad() and the repository sdpa_backend context. Select
'Tile20_after' before each video. First inference compiles/captures CUDA graphs.
The native module targets A100/sm80 and the existing480x832/45-frame runtime.
See Opt24 REPORT.txt for measured quality changes; this is not bit-identical.
"""
from ..opt21.rollout import trajectory

def load():
 from ..opt22.inference import load as parent
 from .pipeline import install
 net,metadata=parent();install()
 return net,metadata
