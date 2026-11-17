"""CUDA Graph of the unchanged latest FP32 route; explicit per-call staging.
No GainVector64 routing, coefficients, thresholds, or proxy features are used.
"""
import torch
from .ops import Summary,RouteOps
from ..gap_route_staging import stage
class GraphRoute:
 def __init__(self,probes=16):
  self.probes=probes;self.entries={};self.pool=torch.cuda.graph_pool_handle();self.calls=0
 @torch.no_grad()
 def __call__(self,q,s,theta):
  key=(tuple(q.shape),s.keys.shape[1]);p=key[1];h,n,d=q.shape
  if key not in self.entries:
   buffers=[torch.empty_like(x, memory_format=torch.contiguous_format) for x in [q,s.keys,s.stats,theta]]
   for dst,src in zip(buffers,[q,s.keys,s.stats,theta]):dst.copy_(src)
   a,k,st,w=buffers
   summary=Summary(k,k[:,:,0],st[:,:,:3,:8],st)
   def run():
    f,_=RouteOps.features(a,summary,check_valid=False)
    return (f*w[:,None,None]).sum(-1).reshape(h,n//self.probes,self.probes,p).mean(2)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(2):warm=run()
   torch.cuda.current_stream().wait_stream(stream);del warm
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream,pool=self.pool):out=run()
   self.entries[key]=(buffers,graph,out)
  buffers,graph,out=self.entries[key]
  empty=q.new_empty(0)
  stage(list(zip(buffers,[q,s.keys,s.stats,theta]))+[(empty,empty)]*5)
  graph.replay();self.calls+=1
  return out.clone()
