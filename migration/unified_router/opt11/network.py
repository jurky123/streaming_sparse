"""Compose Opt10 routing with optional aggressive attention and tuned action heads."""
import os,json,copy
from functools import partial
from pathlib import Path
import torch
from ..common import config,sha
from ..opt10.network import load as parent_load,configure as parent_configure
from .. import fast_network
from .attention import attention

def configure(net,root,variant,trained,base_prepare=None):
 assert variant in ['Compact','Aggressive'];root=Path(root)
 if not hasattr(net,'opt11_initial_heads'):
  net.opt11_initial_heads={key:{k:v.detach().clone()for k,v in head.state_dict().items()}for key,head in net.router.heads.items()if key.endswith('K32-Set')}
 from ..opt7.projection_cache import HistoryProjection
 base_prepare=base_prepare if base_prepare is not None else net.prepare
 while isinstance(base_prepare,HistoryProjection):base_prepare=base_prepare.prepare
 parent_configure(net,'Compact',base_prepare)
 for layer in net.layers:
  key=f'{layer}_K32-Set';head=net.router.heads[key]
  if trained:
   f=root/f'strong_training/layer{layer:02d}/{variant}/best.pt';state=torch.load(f,map_location='cpu',weights_only=True);validation=json.loads((f.parent/'best_validation.json').read_text());assert state['best_metric']=='actual_deployment_attention_nmse'and state['best_value']==validation['nmse'];head.load_state_dict(state['head'])
  else:head.load_state_dict(net.opt11_initial_heads[key])
 if variant=='Aggressive':fast_network.sparse_attention=partial(attention,threshold=0.,pvfp16=True,pack_v=True,fuse_pack=True,splits=2,warps=2,stages=2)
 net.route_cache={};net.opt11_variant=variant;net.opt11_trained=trained;return net

def load(directory,training_seed=0,**options):
 root,m=config(directory);net,_=parent_load(m['parent_experiment'],training_seed,**options);configure(net,root,os.environ.get('SLIMAV_OPT11','Aggressive'),os.environ.get('SLIMAV_OPT11_TRAINED','1')=='1');return net,m

def main():
 from .. import fast_bench
 fast_bench.load_fast=load;fast_bench.main()
if __name__=='__main__':main()
