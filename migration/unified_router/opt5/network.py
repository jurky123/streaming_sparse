"""Opt5 inference adapter; preserves per-query budgets and frozen weights."""
from ..opt2.network import load_optimized
from ..opt4.network import CompactCache
from .weight_route import WeightRoute
from .select import FusedSelectMixin

class OptimizedRoute(FusedSelectMixin,WeightRoute):
    pass

# Chosen using real-block timing, before video validation.
OPTIONS=dict(qm=64,gp=4,energy='content')

def load_flash(directory,training_seed=0,**options):
    net,m=load_optimized(directory,training_seed,**options)
    route=OptimizedRoute(**OPTIONS,batch=net.route_batch)
    net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
    net.compact_options=dict(OPTIONS)
    return net,m

def main():
    from .. import fast_bench
    fast_bench.load_fast=load_flash;fast_bench.main()

if __name__=='__main__':main()
