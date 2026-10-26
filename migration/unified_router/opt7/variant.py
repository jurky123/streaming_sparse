"""Explicitly named Opt7 ablations, never silently replace the accepted model."""
import os
from functools import partial
from ..opt6_guarded import load_guarded_head
from .score_only import load_score
from .reuse import ReuseNetwork
from .split_attention import attention

def load_variant(directory,training_seed=0,**options):
    variant=os.environ.get('SLIMAV_VARIANT','GuardSplit')
    net,m=(load_score if variant.startswith('Score') else load_guarded_head)(directory,training_seed,**options)
    if 'Compact' in variant:
        from .compact_score import CompactScore
        from ..opt4.network import CompactCache
        route=CompactScore();net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
    if 'Cached' in variant:
        from .projection_cache import HistoryProjection
        net.prepare=HistoryProjection(net.prepare)
    if 'Reuse' in variant:
        net.__class__=ReuseNetwork
        policy=variant.split('Reuse')[1].replace('Split','').replace('Counters','')
        net.reuse_policy=policy;net.refresh_steps=tuple(int(v)for v in policy);net.route_cache={}
        assert 0 in net.refresh_steps
    if 'Split' in variant:
        from .. import fast_network
        fast_network.sparse_attention=partial(attention,splits=2,warps=4,stages=2)
    if 'Counters' in variant:
        from .fused_network import FusedNetwork
        net.__class__=FusedNetwork
        if not hasattr(net,'reuse_policy'):
            net.reuse_policy='0123';net.refresh_steps=(0,1,2,3);net.route_cache={}
    return net,m

def main():
    from .. import fast_bench
    fast_bench.load_fast=load_variant;fast_bench.main()
if __name__=='__main__':main()
