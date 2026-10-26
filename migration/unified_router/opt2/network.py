"""Independent second-stage adapter. No shared-query routing is accepted."""
from functools import partial
import torch
from ..fast_network import load_fast
from ..fast_route import FastRoute
from ..fast_graph import GraphCache
from .features import split_finish
from .score import direct_score
from .candidates import candidate_sides_fast
from .attention import attention

def make_route(batch=512):
    route=FastRoute(True,True,batch=batch)
    route.score=partial(direct_score,tile=32)
    route.finish=split_finish
    route.candidates=torch.compile(candidate_sides_fast,fullgraph=True,dynamic=False)
    return route


def load_optimized(directory,training_seed=0,**options):
    if options.get('query_stride',1)!=1:raise ValueError('opt2 preserves every query decision; query_stride must be 1')
    options.update(graph=True,fused=True,history_cache=True,compiled_build=True,query_stride=1,group=32)
    net,m=load_fast(directory,training_seed,**options)
    # Scheduling groups 32 query rows, but retains each row's individual mask.
    # This does not broadcast/reuse any query's routing decision.
    from .. import fast_network
    fast_network.sparse_attention=partial(attention,warps=4,stages=3)
    route=make_route(net.route_batch)
    net.fast_route=route;net.graph_route=GraphCache(route,net.route_batch)
    return net,m
