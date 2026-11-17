"""Replay unchanged latest summary equations; stage every input and layer weight."""
import copy
import torch
from .ops import Summary,SummaryOps,haar
class GraphSummary:
 def __init__(self):
  self.entries={};self.calls=0;self.pool=torch.cuda.graph_pool_handle()
 @torch.no_grad()
 def __call__(self,model,k,v,cheap=False):
  key=(cheap,tuple(k.shape),k.stride(),v.stride(),k.dtype,k.device)
  if key not in self.entries:
   kk=torch.empty_strided(k.shape,k.stride(),dtype=k.dtype,device=k.device);vv=torch.empty_strided(v.shape,v.stride(),dtype=v.dtype,device=v.device)
   mm=copy.deepcopy(model).requires_grad_(False);kk.copy_(k);vv.copy_(v);ha=haar(k)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):SummaryOps.build(mm,kk,vv,cheap,haar_basis=ha)
   torch.cuda.current_stream().wait_stream(stream)
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,pool=self.pool):result=SummaryOps.build(mm,kk,vv,cheap,haar_basis=ha)
   self.entries[key]=(kk,vv,mm,graph,result,ha)
  kk,vv,mm,graph,result,ha=self.entries[key];kk.copy_(k);vv.copy_(v)
  for name,param in model.named_parameters():mm.get_parameter(name).copy_(param)
  graph.replay();self.calls+=1
  # Results cannot alias the graph's next invocation or a later clean commit.
  return Summary(*(getattr(result,n).clone() for n in ('keys','mean','detail','stats')))
