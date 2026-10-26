"""Compact baseline with associative candidate reduction; no voting."""
import os
from ..common import config
from ..opt10.network import load as base_load
from ..opt13.network import configure as baseline_configure
from ..opt10.compact_route import CompactMomentRoute
from ..opt4.network import CompactCache
from .select import select
from .setup import PARENT
class Top3Route(CompactMomentRoute):
 def run(self,layer,q,sketch,state,step,router):
  assert router.kind=='K32-Set'
  score,raw=self.score(q,state);mask,rem,add=select(score,sketch.origins,router.beta,warps=1)
  stats,cv=self.stats(q,sketch,score,mask,rem,add,raw,state)
  return self.actions(q,sketch,score,mask,rem,add,router.beta,step,stats,cv,router.heads[f'{layer}_{router.kind}'])
def configure(net,root,variant='Select',base_prepare=None):
 baseline_configure(net,PARENT,'Reference',base_prepare)
 if variant=='Select':
  route=Top3Route(batch=4680);net.fast_route=route;net.graph_route=CompactCache(route,4680)
 elif variant!='Reference':raise ValueError(variant)
 return net
def load(directory,training_seed=0,**options):
 root,m=config(directory);net,_=base_load(m['parent_experiment'],training_seed,**options);configure(net,root,os.environ.get('SLIMAV_OPT14','Select'));return net,m
