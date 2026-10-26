"""Full-router optimization and explicitly selected quantized-attention ablations."""
import os
from functools import partial
import torch,triton as tr
from ..opt6_guarded import GuardedHeadRoute,load_guarded_head
from ..opt5.weight_route import _weighted_candidates
from ..opt4.kernels import _gram_stats
from ..opt4.network import CompactCache
from ..opt7.fused_network import FusedNetwork
from ..opt7.projection_cache import HistoryProjection
from ..opt7.compact_score import CompactScore
from ..opt7.split_attention import attention as bf16_attention
from .. import fast_network
from .dual_means import means
from .block_int8_attention import attention as qk_attention
from .int8_pv_attention import attention as pv_attention

class DualMeanRoute(GuardedHeadRoute):
 def stats(self,q,sketch,score,mask,rem,add,raw,state):
  weights,energy=raw;h,n,b=score.shape;r=weights.shape[1]
  stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
  cv=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
  mean=means(weights,mask,state,bm=32,bk=64,warps=4,stages=3)
  _weighted_candidates[(h*n,)](weights,state['value'],state['starts'],state['counts'],sketch.costs,rem,add,cv,n,r,num_warps=4,enable_fp_fusion=True)
  _gram_stats[(h*n,)](mean,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),num_warps=4,enable_fp_fusion=False)
  return stats,cv

def configure(net,variant,base_prepare=None):
 assert variant in ['Opt7Full','Full','QK','PV','QKFast','PVFast','Proj','QKProj','Opt7Fast']
 net.__class__=FusedNetwork
 route=CompactScore()if variant.endswith('Fast')else GuardedHeadRoute(batch=net.route_batch)if variant=='Opt7Full'else DualMeanRoute(batch=net.route_batch)
 if 'Proj' in variant:
  from .projected_route import ProjectedRoute
  route=ProjectedRoute(batch=net.route_batch)
 net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch)
 net.prepare=HistoryProjection(base_prepare if base_prepare is not None else net.prepare)
 net.reuse_policy='0'if variant.endswith('Fast')else'0123';net.refresh_steps=tuple(map(int,net.reuse_policy));net.route_cache={}
 kernel=pv_attention if variant.startswith('PV')else qk_attention if variant.startswith('QK')else bf16_attention
 fast_network.sparse_attention=partial(kernel,splits=2,warps=2 if variant.startswith(('QK','PV'))else 4,stages=2)
 return net

def load(directory,training_seed=0,**options):
 net,m=load_guarded_head(directory,training_seed,**options)
 configure(net,os.environ.get('SLIMAV_OPT8','Full'));return net,m

def main():
 from .. import fast_bench
 fast_bench.load_fast=load;fast_bench.main()
if __name__=='__main__':main()
