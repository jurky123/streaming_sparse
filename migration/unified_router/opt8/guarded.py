"""Retain BF16 attention in the final layer after quantization audit failure."""
import os
from functools import partial
from ..opt7.fused_network import FusedNetwork
from ..opt7.split_attention import attention as bf16_attention
from ..opt6_guarded import load_guarded_head
from .. import fast_network
from .network import configure
class PrecisionNetwork(FusedNetwork):
 def _block(self,block,layer,*args,**kwargs):
  self.attention_layer=layer
  return super()._block(block,layer,*args,**kwargs)
class LayerAttention:
 def __init__(self,net,quantized):
  self.net=net;self.quantized=quantized;self.reference=partial(bf16_attention,splits=2,warps=4,stages=2)
 def __call__(self,*args,**kwargs):
  return (self.reference if self.net.attention_layer==29 else self.quantized)(*args,**kwargs)
def configure_guarded(net,variant,base_prepare=None):
 assert variant in ['QKProjGuard','QKFastGuard']
 configure(net,variant.removesuffix('Guard'),base_prepare)
 net.__class__=PrecisionNetwork;net.attention_layer=-1
 fast_network.sparse_attention=LayerAttention(net,fast_network.sparse_attention)
 return net
def load(directory,training_seed=0,**options):
 net,m=load_guarded_head(directory,training_seed,**options)
 configure_guarded(net,os.environ.get('SLIMAV_OPT8','QKProjGuard'));return net,m
