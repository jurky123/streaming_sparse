"""Frozen Tile20 runtime, adding exact BF16 pointwise/RMS and QKV fusion.

Call under torch.no_grad and the existing sdpa_backend context. Select
'Tile20_after' once before every video to refresh conditioning/cache ownership.
The ordered RMS implementation targets the pinned PyTorch CUDA reduction order
for BF16 1536-channel tensors with at least 16 rows. First rollout builds graphs.
"""
from ..opt21.rollout import trajectory

def load():
 from ..opt21.inference import load as parent_load
 from .network import configure
 net,metadata=parent_load();configure(net,rms_mode='ordered',qkv=True)
 return net,metadata
