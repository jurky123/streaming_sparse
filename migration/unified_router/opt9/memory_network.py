import os
from functools import partial
from ..opt6_guarded import load_guarded_head
from .. import fast_network
from .network import configure as previous_configure
from .segmented_network import SegmentedNetwork
from .segmented_attention import attention

def configure(net,variant,base_prepare=None):
 previous_configure(net,{'MemoryFull':'Packed','MemoryFast':'Opt8Fast'}.get(variant,variant),base_prepare)
 if variant.startswith('Memory'):
  net.__class__=SegmentedNetwork
  fast_network.sparse_attention=partial(attention,splits=2,warps=2,stages=2,pack_v=True,fuse_pack=True)
 return net

def load(directory,training_seed=0,**options):
 net,m=load_guarded_head(directory,training_seed,**options);configure(net,os.environ.get('SLIMAV_OPT9','MemoryFull'));return net,m

def main():
 from .. import fast_bench
 fast_bench.load_fast=load;fast_bench.main()
if __name__=='__main__':main()
