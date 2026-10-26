"""Capture an entire denoising step, with explicit trajectory-owned input refresh."""
import torch
from types import SimpleNamespace
from streaming_sparse.v08.cache import LayerKV
from .runtime import RuntimeNetwork
from ..opt20.network import TileNetwork
from .runtime import PipelineState

class StepGraph:
 def __init__(self,net,latent,embeds,timestep,cache,start,sparse,identity):
  self.net=net;self.latent=torch.empty_like(latent);self.latent.copy_(latent)
  self.t=torch.full(latent.shape[:2],float(timestep),device=latent.device,dtype=torch.float32)
  e,e0=net.time_features(timestep);self.e=e.clone();self.e0=e0.clone()
  self.cache=tuple(LayerKV(x.k,x.v,identity,x.layer) for x in cache)
  self.token=None;self.graph=torch.cuda.CUDAGraph();self.sparse=sparse;self.start=start
  def run():
   net.runtime_timestep=self.t;net.runtime_time_features=(self.e,self.e0);net.runtime_full_step=True;previous=net.routing_records;net.routing_records=[]
   result=RuntimeNetwork.forward_chunk(net,self.latent,embeds,timestep,self.cache,start,sparse=sparse,step=0,identity=identity,checkpoint=False,collect_aux=False,write_cache=False)
   self.records=list(net.routing_records);net.routing_records=previous
   stats=torch.stack([torch.stack((r['retained_H'],r['retained_C'])) for r in self.records]) if self.records else None
   return result,stats
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(2):run()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  with torch.cuda.graph(self.graph,pool=net.step_pool):self.output,self.stats=run()
  net.runtime_timestep=None;net.runtime_time_features=None;net.runtime_full_step=False
 def __call__(self,latent,timestep,cache,identity,step,current_start):
  token=(self.net.tile_epoch,identity,tuple((x.k.data_ptr(),x.k._version) for x in cache))
  if token!=self.token:
   for old,new in zip(self.cache,cache):
    assert old.k.data_ptr()==new.k.data_ptr() and old.v.data_ptr()==new.v.data_ptr(), 'History must use the bounded shared workspace'
   if self.sparse and self.cache:
    for layer in self.net.layers:
     raw=self.cache[layer];plan=self.net.tile_graphs[(layer,raw.k.shape[1],3*(latent.shape[-2]//2)*(latent.shape[-1]//2))]
     plan.refresh_history(raw.k[0].transpose(0,1),raw.v[0].transpose(0,1),token)
   self.token=token
  self.latent.copy_(latent);self.t.fill_(float(timestep));e,e0=self.net.time_features(timestep);self.e.copy_(e);self.e0.copy_(e0);self.graph.replay()
  if self.stats is not None:
   stats=self.stats.clone()
   for i,r in enumerate(self.records):self.net.routing_records.append(dict(r,start=current_start,step=step,retained_H=stats[i,0],retained_C=stats[i,1]))
  return self.output

class GraphNetwork(RuntimeNetwork):
 def time_features(self,timestep):
  from wan.modules.model import sinusoidal_embedding_1d
  key=float(timestep)
  if key not in self.runtime_times:
   t=torch.full((1,3),key,device='cuda',dtype=torch.float32)
   e=self.model.time_embedding(sinusoidal_embedding_1d(self.model.freq_dim,t.flatten()).to(torch.bfloat16))
   e0=self.model.time_projection(e).unflatten(1,(6,self.model.dim)).unflatten(0,t.shape)
   self.runtime_times[key]=(e,e0)
  return self.runtime_times[key]
 def select(self,name):
  mode=name!='Dense'
  if getattr(self,'step_mode',None)!=mode:
   self.step_graphs.clear();self.step_pool=torch.cuda.graph_pool_handle()
   import gc
   gc.collect();torch.cuda.empty_cache()
  self.step_mode=mode
  if name!='Dense' and self.tile_active not in (None,name):
   self.step_graphs.clear();self.step_pool=torch.cuda.graph_pool_handle()
  self.needs_condition=True
  return TileNetwork.select(self,name)
 def condition(self,embeds):
  context=self.runtime_text_forward(torch.nn.functional.pad(embeds,(0,0,0,self.model.text_len-embeds.shape[1])))
  if self.runtime_context is None:self.runtime_context=context
  else:self.runtime_context.copy_(context)
  for block,cache in zip(self.model.blocks,self.runtime_cross):
   att=block.cross_attn;b=embeds.shape[0];k=att.norm_k(att.k(context)).view(b,-1,att.num_heads,att.head_dim);v=att.v(context).view_as(k)
   if cache['is_init']:cache['k'].copy_(k);cache['v'].copy_(v)
   else:cache.update(is_init=True,k=k,v=v)
  self.needs_condition=False
 def forward_chunk(self,latent,embeds,timestep,cache,start,sparse=True,step=0,keep=.2,identity=None,
                   checkpoint=True,collect_aux=True,detach_router=True,mass_only=False,collect_mechanisms=False,write_cache=True):
  self.rope_start.fill_(start)
  if self.needs_condition:self.condition(embeds)
  self.runtime_timestep=None;self.runtime_time_features=None
  if write_cache:
   return super().forward_chunk(latent,embeds,timestep,cache,start,sparse,step,keep,identity,checkpoint,collect_aux,detach_router,mass_only,collect_mechanisms,True)
  assert not collect_aux and not torch.is_grad_enabled()
  refresh=True
  if sparse and cache and getattr(self,'active_options',None) is not None:
   from ..opt30.setup import source_step
   refresh=source_step(step,self.active_options['refresh'])==step
  self.aggregation_refresh=refresh
  key=(sparse,start if getattr(self,'static_position_graphs',False) else min(start,9),refresh)
  if key not in self.step_graphs:self.step_graphs[key]=StepGraph(self,latent,embeds,timestep,cache,start,sparse,identity)
  return self.step_graphs[key](latent,timestep,cache,identity,step,start)
