"""Frozen Q32/20% inference on A100; 480p/720p, four chunks (45 RGB frames).

Use under torch.no_grad and sdpa_backend. Select Tile20_after for each video.
The first rollout builds CUDA graphs. Runtime owns one sequential trajectory;
shared history workspaces must not be used concurrently by multiple streams.
"""
from ..opt25.rollout import trajectory as _trajectory

def trajectory(*args,**kwargs):
 if kwargs.get("num_chunks",4)!=4:raise ValueError("This runtime supports four chunks / 45 RGB frames")
 return _trajectory(*args,**kwargs)

def load():
 from ..opt22.inference import load as parent_load
 from ..opt25.network import FusedNetwork
 from ..opt25.pipeline import PipelineState
 from .pipeline import install
 net,metadata=parent_load();net.__class__=FusedNetwork;net.history_chunks=3
 PipelineState.workspaces.clear();PipelineState.sparge=True;install()
 return net,metadata
