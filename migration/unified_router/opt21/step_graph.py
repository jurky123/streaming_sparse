"""Capture an entire denoising step, with explicit trajectory-owned input refresh."""
import torch
from types import SimpleNamespace
from streaming_sparse.v08.cache import LayerKV
from .network import RuntimeNetwork,configure
from ..opt20.network import TileNetwork,TileGraph

class PipelineState(TileGraph):
 """Own KV and prototypes only; the outer graph captures the actual kernels."""
 def __init__(self,model,raw):
  from ..opt20.network import Pipeline
  self.pipe=Pipeline(model,raw,32,model.budget,'shared','fp16')
  self.pipe.cache.history=SimpleNamespace(parts=self.pipe.cache.history.parts)
  self.identity=None

class StepGraph:
 def __init__(self,net,latent,embeds,timestep,cache,start,sparse,identity):
  self.net=net;self.latent=torch.empty_like(latent);self.latent.copy_(latent)
  self.t=torch.full(latent.shape[:2],float(timestep),device=latent.device,dtype=torch.float32)
  e,e0=net.time_features(timestep);self.e=e.clone();self.e0=e0.clone()
  self.cache=tuple(LayerKV(x.k.clone(),x.v.clone(),identity,x.layer) for x in cache)
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
 def __call__(self,latent,timestep,cache,identity,step):
  token=(self.net.tile_epoch,identity,tuple((x.k.data_ptr(),x.k._version) for x in cache))
  if token!=self.token:
   for old,new in zip(self.cache,cache):old.k.copy_(new.k);old.v.copy_(new.v)
   if self.sparse and self.cache:
    for layer in self.net.layers:
     raw=self.cache[layer];plan=self.net.tile_graphs[(layer,raw.k.shape[1],4680)]
     plan.refresh_history(raw.k[0].transpose(0,1),raw.v[0].transpose(0,1),token)
   self.token=token
  self.latent.copy_(latent);self.t.fill_(float(timestep));e,e0=self.net.time_features(timestep);self.e.copy_(e);self.e0.copy_(e0);self.graph.replay()
  if self.stats is not None:
   stats=self.stats.clone()
   for i,r in enumerate(self.records):self.net.routing_records.append(dict(r,step=step,retained_H=stats[i,0],retained_C=stats[i,1]))
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
  if self.needs_condition:self.condition(embeds)
  self.runtime_timestep=None;self.runtime_time_features=None
  if write_cache:
   return super().forward_chunk(latent,embeds,timestep,cache,start,sparse,step,keep,identity,checkpoint,collect_aux,detach_router,mass_only,collect_mechanisms,True)
  assert not collect_aux and not torch.is_grad_enabled()
  key=(sparse,start)
  if key not in self.step_graphs:self.step_graphs[key]=StepGraph(self,latent,embeds,timestep,cache,start,sparse,identity)
  return self.step_graphs[key](latent,timestep,cache,identity,step)

 def _block(self,block,layer,x,e0,grid,freqs,context,history_k,history_v,start,sparse,step,keep,
            collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity):
     if not sparse or layer not in self.layers or not history_k.shape[1] or self.router.kind=='Dense':
         return self._dense_block(block,layer,x,e0,grid,freqs,context,history_k,history_v,start,False,step,keep,
             collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity)
     if torch.is_grad_enabled() or collect_aux:raise RuntimeError('Opt20 video adapter is evaluation-only')
     from wan.modules.causal_model import causal_rope_apply
     frames=e0.shape[1];tokens=x.shape[1]//frames
     e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
     mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2)
     att=block.self_attn;b,s=mod.shape[:2];h=att.num_heads;d=att.head_dim
     q=att.norm_q(att.q(mod)).view(b,s,h,d);k=att.norm_k(att.k(mod)).view(b,s,h,d);v=att.v(mod).view(b,s,h,d)
     q=causal_rope_apply(q,grid,freqs,start_frame=start).type_as(v);k=causal_rope_apply(k,grid,freqs,start_frame=start).type_as(v)
     qq=q[0].transpose(0,1);raw=dict(kh=history_k[0].transpose(0,1),vh=history_v[0].transpose(0,1),kc=k[0].transpose(0,1),vc=v[0].transpose(0,1))
     key=(layer,raw['kh'].shape[1],s)
     if key not in self.tile_graphs:self.tile_graphs[key]=PipelineState(self.tile_models[str(layer)],raw)
     identity=(self.tile_epoch,cache_identity,history_k.data_ptr(),history_v.data_ptr(),history_k._version,history_v._version)
     plan=self.tile_graphs[key]
     if getattr(self,'runtime_full_step',False):
         if plan.identity!=identity:plan.refresh_history(raw['kh'],raw['vh'],identity)
         y,aux=plan.pipe.run(qq,raw['kc'],raw['vc'],True);ids=aux['selection']
     else:y,ids=plan(qq,raw,identity)
     y=y.transpose(0,1)[None]
     x=x+(att.o(y.flatten(2)).unflatten(1,(frames,tokens))*e[2]).flatten(1,2)
     x=x+block.cross_attn(block.norm3(x),context,None,crossattn_cache=None)
     ff=block.ffn((block.norm2(x).unflatten(1,(frames,tokens))*(1+e[4])+e[3]).flatten(1,2))
     x=x+(ff.unflatten(1,(frames,tokens))*e[5]).flatten(1,2)
     nh=raw['kh'].shape[1];nq=h*s
     rows=(s-torch.arange(ids.shape[1],device=ids.device)*32).clamp(max=32).float()
     cost=torch.where(ids%74==73,8.,64.)*rows[None,:,None]
     rh=torch.where(ids<nh//4680*74,cost,0.).sum();rc=torch.where(ids>=nh//4680*74,cost,0.).sum()
     executed=ids.numel()*32*64;zero=x.new_zeros((),dtype=torch.float32)
     self.routing_records.append(dict(layer=layer,start=start,step=step,queries=nq,history_tokens=nh,current_tokens=s,
         retained_H=rh,retained_C=rc,executed_pairs=executed,keep=(rh+rc)/(nq*(nh+s))))
     diagnostic=torch.stack((rh+rc,rh.new_full((),nq),rh,rc,rh.new_full((),nq*(nh+s)),rh.new_full((),1)))
     return x,k,v,zero,zero,diagnostic,zero,[]
 def _dense_block(self,block,layer,x,e0,grid,freqs,context,history_k,history_v,start,sparse,step,keep,
            collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity):
     from wan.modules.causal_model import causal_rope_apply
     frames=e0.shape[1];tokens=x.shape[1]//frames
     e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
     mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2)
     att=block.self_attn;b,s=mod.shape[:2];h=att.num_heads;d=att.head_dim
     q=att.norm_q(att.q(mod)).view(b,s,h,d)
     k=att.norm_k(att.k(mod)).view(b,s,h,d);v=att.v(mod).view(b,s,h,d)
     q=causal_rope_apply(q,grid,freqs,start_frame=start).type_as(v)
     k=causal_rope_apply(k,grid,freqs,start_frame=start).type_as(v)
     aux_att=x.new_zeros((),dtype=torch.float32);aux_cal=aux_att.clone()
     # Counters are returned, never appended to shared checkpoint state.
     diagnostic=x.new_zeros(6,dtype=torch.float32);aux_count=0;records=[]
     if not sparse or layer not in self.layers or not history_k.shape[1] or self.router is None:
         all_k=torch.cat((history_k,k),1);all_v=torch.cat((history_v,v),1)
         y=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),all_k.transpose(1,2),
             all_v.transpose(1,2),dropout_p=0.,is_causal=False).transpose(1,2)
     x=x+(att.o(y.flatten(2)).unflatten(1,(frames,tokens))*e[2]).flatten(1,2)
     # No cross-attention mutable cache: recomputation is pure.
     x=x+block.cross_attn(block.norm3(x),context,None,crossattn_cache=None)
     ff=block.ffn((block.norm2(x).unflatten(1,(frames,tokens))*(1+e[4])+e[3]).flatten(1,2))
     x=x+(ff.unflatten(1,(frames,tokens))*e[5]).flatten(1,2)
     return x,k,v,aux_att,aux_cal,diagnostic,x.new_full((),float(aux_count),dtype=torch.float32),records


def configure_graph(net,kernel=False):
 configure(net,kernel);net.__class__=GraphNetwork;net.step_graphs={};net.step_pool=torch.cuda.graph_pool_handle();net.needs_condition=True
 net.runtime_times={};net.runtime_freqs=net.model.freqs.to('cuda')
 net.runtime_scheduler=(net.wrapper.scheduler.sigmas.double().cuda(),net.wrapper.scheduler.timesteps.double().cuda())
 return net
