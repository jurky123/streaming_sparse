"""Compact-only adapter; accepted baseline is Opt11 head-tuned Compact."""
import os
from ..common import config
from ..opt10.network import load as base_load
from ..opt11.network import configure as baseline_configure

def configure(net,root,variant='Reference',base_prepare=None):
 root,m=config(root)
 baseline_configure(net,m['parent_opt11'],'Compact',True,base_prepare)
 if variant.startswith('FusedVote'):
  assert len(net.refresh_steps)==4, 'Fused union metadata requires routing refresh on every step'
  from .fused_vote import FusedVotingRoute
  from .fused_attention import attention
  from .. import fast_network
  from functools import partial
  net.graph_route=FusedVotingRoute(net.graph_route,int(variant[9:]))
  fast_network.sparse_attention=partial(attention,pack_v=True,fuse_pack=True)
 elif variant.startswith('Vote'):
  from .vote import VotingRoute
  route=VotingRoute(net.graph_route,int(variant[4:]))
  net.graph_route=route
 net.opt13_variant=variant
 return net

def load(directory,training_seed=0,**options):
 root,m=config(directory);net,_=base_load(m['parent_experiment'],training_seed,**options)
 configure(net,root,os.environ.get('SLIMAV_OPT13','Reference'))
 return net,m
