def load():
 from ..opt22.inference import load as parent
 from .network import FusedNetwork
 from .runtime import PipelineState
 net,m=parent();net.__class__=FusedNetwork;net.history_chunks=3;net.aggregation_refresh=True;net.active_options=None
 PipelineState.workspaces.clear();return net,m
