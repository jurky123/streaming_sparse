"""Opt6 head-only adapter; Opt5 scoring, candidates and attention are retained."""
from ..opt2.network import load_optimized
from ..opt4.network import CompactCache
from .route import HeadRoute
OPTIONS=dict(method='factor',warps=1,pair='tf32x3')

def load_head(directory,training_seed=0,**options):
    net,m=load_optimized(directory,training_seed,**options)
    route=HeadRoute(**OPTIONS,batch=net.route_batch)
    net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
    net.head_options=dict(OPTIONS)
    return net,m

def main():
    from .. import fast_bench
    fast_bench.load_fast=load_head;fast_bench.main()

if __name__=='__main__':main()
