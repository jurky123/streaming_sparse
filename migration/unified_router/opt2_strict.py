"""Conservative ablation: retain Fast1 score math and feature reduction width."""
from functools import partial
import torch
from .opt2.network import load_optimized
from .opt2.features import split_finish
from .fast_route import score_cached
from .fast_graph import GraphCache

def load_strict(directory,training_seed=0,**options):
    net,m=load_optimized(directory,training_seed,**options)
    net.fast_route.score=torch.compile(score_cached,fullgraph=True,dynamic=False)
    net.fast_route.finish=partial(split_finish,stat_warps=8,action_warps=8)
    net.graph_route=GraphCache(net.fast_route,net.route_batch)
    return net,m

def main():
    from . import fast_bench
    fast_bench.load_fast=load_strict;fast_bench.main()

if __name__=='__main__':main()
