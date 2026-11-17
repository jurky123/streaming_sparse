"""Graph replay of unchanged selection and physical grouping operations."""
import copy,torch
from .ops import RouteOps
from .runtime import GroupingOps
class GraphSelection:
 def __init__(self,config):self.config=config;self.entries={};self.calls=0;self.pool=torch.cuda.graph_pool_handle()
 @torch.no_grad()
 def __call__(self,scores):
  h,t,p=scores.shape;assert (h,t)==(12,72) and p>=72
  key=(h,t,p,scores.device,scores.dtype)
  if key not in self.entries:
   ss=torch.empty_like(scores);ss.copy_(scores)
   own=(torch.arange(t,device=scores.device)+p-72)[None].expand(h,-1)
   # All pages are valid; own is constructed in [P-72,P), so no device sync is needed.
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):RouteOps.select(ss,own,self.config,check_valid=False)
   torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,pool=self.pool):out=RouteOps.select(ss,own,self.config,check_valid=False)
   self.entries[key]=(ss,own,graph,out)
  ss,own,graph,out=self.entries[key];ss.copy_(scores);graph.replay();self.calls+=1
  return {n:v.clone() for n,v in out.items()}
class GraphGrouping:
 def __init__(self,config):self.config=config;self.entries={};self.calls=0;self.pool=torch.cuda.graph_pool_handle()
 @torch.no_grad()
 def __call__(self,model,k,v):
  key=(tuple(k.shape),k.stride(),v.stride(),k.device,k.dtype)
  if key not in self.entries:
   kk=torch.empty_strided(k.shape,k.stride(),device=k.device,dtype=k.dtype);vv=torch.empty_strided(v.shape,v.stride(),device=v.device,dtype=v.dtype)
   kk.copy_(k);vv.copy_(v);mm=copy.deepcopy(model).requires_grad_(False);order=self.config.page_order(k.device)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):GroupingOps.group(mm,kk,vv,self.config,page_order=order)
   torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,pool=self.pool):out=GroupingOps.group(mm,kk,vv,self.config,page_order=order)
   self.entries[key]=(kk,vv,mm,order,graph,out)
  kk,vv,mm,order,graph,out=self.entries[key];kk.copy_(k);vv.copy_(v)
  for n,param in model.named_parameters():mm.get_parameter(n).copy_(param)
  graph.replay();self.calls+=1
  return tuple(x.clone() for x in out)
