def load():
 from ..opt22.inference import load as parent
 from .network import FusedNetwork
 from .runtime import PipelineState
 net,m=parent();net.__class__=FusedNetwork;net.history_chunks=3;net.aggregation_refresh=True;net.active_options=None
 import torch
 from . import rope
 net.rope_start=torch.zeros((),device="cuda",dtype=torch.int32);rope.POSITION=net.rope_start
 PipelineState.workspaces.clear();return net,m
