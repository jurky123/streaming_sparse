"""Populate fixed-shape compiler caches for the required policy head counts, no tuning."""
import argparse,json,time
import torch
from .weights import ROOT
from .deployment import current,configure,load_weights
from .pipeline_graph import BatchGraphs
from .risk_projected import risk
from ..review_v1.ops import Config,Summary
from . import pipeline_graph
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--pages',type=int,nargs='+',default=[432,360,288,216]);p.add_argument('--counts',type=int,nargs='+',required=True);a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.cache_size_limit=64;torch._dynamo.config.accumulated_cache_size_limit=512;configure(current());pipeline_graph.CACHED_LOGMASS=True
 _,_,phi,_,_=load_weights(current());saved=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
 q=saved['q'].cuda().index_select(2,saved['offsets'].cuda());theta=saved['theta'].cuda();basis=saved['basis'].cuda()
 rt=BatchGraphs(risk,phi,Config(),compile_route=True,compile_tail=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True,input_m16=True,static_route=True)
 for pages in a.pages:
  s=Summary(*(saved['summary'][n][:,:pages].cuda() for n in ('keys','mean','detail','stats')));en={k:(v[:,:pages].cuda().bfloat16() if k in ('co','A') else v[:,:pages].cuda()) for k,v in saved['en'].items()};own=(torch.arange(72,device='cuda')+pages-72)[None].expand(12,-1)
  for count in a.counts:
   t=time.perf_counter();rt.get(q,s,en,theta,basis,own,True,heads=torch.arange(count,device='cuda'));print(pages,count,time.perf_counter()-t,flush=True)
if __name__=='__main__':main()
