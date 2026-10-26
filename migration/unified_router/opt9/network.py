"""Opt9: compact candidate keys and online PV skipping, with Opt8 controls."""
import os
from functools import partial
from ..opt8.network import configure as prior_configure
from ..opt8.projected_route import ProjectedRoute
from ..opt4.network import CompactCache
from ..opt6_guarded import load_guarded_head
from .. import fast_network
from .select import fused_select
from .skip_attention import attention
class PackedRoute(ProjectedRoute):
 def run(self,layer,q,sketch,state,step,router):
  if router.kind=='U-Sol':return super().run(layer,q,sketch,state,step,router)
  score,raw=self.score(q,state)
  mask,rem,add=fused_select(score,sketch.origins,router.beta,mode='packed32')
  stats,cv=self.stats(q,sketch,score,mask,rem,add,raw,state)
  return self.actions(q,sketch,score,mask,rem,add,router.beta,step,stats,cv,router.heads[f'{layer}_{router.kind}'])
def configure(net,variant,base_prepare=None):
 assert variant in ['Opt8Full','Opt8Fast','Packed','Skip2','Skip0','Skip2Fast','Skip0Fast']
 prior_configure(net,'QKFast'if variant.endswith('Fast')else'QKProj',base_prepare)
 if variant not in ['Opt8Full','Opt8Fast']and not variant.endswith('Fast'):
  route=PackedRoute(batch=net.route_batch);net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
 if variant.startswith('Skip'):
  fast_network.sparse_attention=partial(attention,threshold=-2. if variant.startswith('Skip2')else 0.,splits=2,warps=2,stages=2)
 return net
def load(directory,training_seed=0,**options):
 net,m=load_guarded_head(directory,training_seed,**options);configure(net,os.environ.get('SLIMAV_OPT9','Skip2'));return net,m
def main():
 from .. import fast_bench
 fast_bench.load_fast=load;fast_bench.main()
if __name__=='__main__':main()
