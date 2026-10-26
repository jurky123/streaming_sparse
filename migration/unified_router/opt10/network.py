"""Whole-layer independent routing and explicitly selected numerical ablations."""
import os
from ..opt9.memory_network import configure as prior_configure
from ..opt9.network import PackedRoute
from ..opt6_guarded import load_guarded_head
from ..opt4.network import CompactCache
from .tuned_route import TunedHeadRoute,TunedMomentRoute
from .quant_route import QuantRoute
from .compact_route import CompactMomentRoute

def configure(net,variant,base_prepare=None):
 assert variant in ['Opt9','Wide','Head','Moments','Compact','Quant']
 prior_configure(net,'MemoryFull',base_prepare)
 if variant!='Opt9':
  cls={'Wide':PackedRoute,'Head':TunedHeadRoute,'Moments':TunedMomentRoute,'Compact':CompactMomentRoute,'Quant':QuantRoute}[variant]
  route=cls(batch=4680);net.fast_route=route;net.graph_route=CompactCache(route,4680)
 return net

def load(directory,training_seed=0,**options):
 net,m=load_guarded_head(directory,training_seed,**options);configure(net,os.environ.get('SLIMAV_OPT10','Head'));return net,m

def main():
 from .. import fast_bench
 fast_bench.load_fast=load;fast_bench.main()
if __name__=='__main__':main()
