"""Opt21 adapter with frozen-backbone operation fusion; budgets/checkpoints unchanged."""
import gc
import torch
from .step_graph import GraphNetwork
from .pipeline import PipelineState
from ..opt22.kernels import modulate,residual,rms,rms_ordered

class FusedNetwork(GraphNetwork):
 def project(self,layer,att,x):
  b,s,_=x.shape;h=att.num_heads;d=att.head_dim
  if self.fused_qkv:
   w,bias=self.qkv_parameters[layer];z=torch.nn.functional.linear(x,w,bias);q,k,v=z.split(h*d,dim=-1)
   return att.norm_q(q).view(b,s,h,d),att.norm_k(k).view(b,s,h,d),v.view(b,s,h,d)
  return att.norm_q(att.q(x)).view(b,s,h,d),att.norm_k(att.k(x)).view(b,s,h,d),att.v(x).view(b,s,h,d)

 def _block(self,block,layer,x,e0,grid,freqs,context,history_k,history_v,start,sparse,step,keep,
            collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity):
     if not sparse or layer not in self.layers or not history_k.shape[1] or self.router.kind=='Dense':
         return self._dense_block(block,layer,x,e0,grid,freqs,context,history_k,history_v,start,False,step,keep,
             collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity)
     if torch.is_grad_enabled() or collect_aux:raise RuntimeError('Opt20 video adapter is evaluation-only')
     from wan.modules.causal_model import causal_rope_apply
     frames=e0.shape[1];tokens=x.shape[1]//frames
     e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
     mod=modulate(block.norm1(x),e[1],e[0])
     att=block.self_attn;b,s=mod.shape[:2];h=att.num_heads;d=att.head_dim
     q,k,v=self.project(layer,att,mod)
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
     x=residual(x,att.o(y.flatten(2)),e[2])
     x=x+block.cross_attn(block.norm3(x),context,None,crossattn_cache=None)
     ff=block.ffn(modulate(block.norm2(x),e[4],e[3]))
     x=residual(x,ff,e[5])
     nh=raw['kh'].shape[1];nq=h*s
     rows=(s-torch.arange(ids.shape[1],device=ids.device)*32).clamp(max=32).float()
     pages=(s+63)//64
     cost=torch.where(ids%pages==pages-1,float(s%64 or 64),64.)*rows[None,:,None]
     rh=torch.where(ids<nh//s*pages,cost,0.).sum();rc=torch.where(ids>=nh//s*pages,cost,0.).sum()
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
     mod=modulate(block.norm1(x),e[1],e[0])
     att=block.self_attn;b,s=mod.shape[:2];h=att.num_heads;d=att.head_dim
     q,k,v=self.project(layer,att,mod)
     q=causal_rope_apply(q,grid,freqs,start_frame=start).type_as(v)
     k=causal_rope_apply(k,grid,freqs,start_frame=start).type_as(v)
     aux_att=x.new_zeros((),dtype=torch.float32);aux_cal=aux_att.clone()
     # Counters are returned, never appended to shared checkpoint state.
     diagnostic=x.new_zeros(6,dtype=torch.float32);aux_count=0;records=[]
     if not sparse or layer not in self.layers or not history_k.shape[1] or self.router is None:
         all_k=torch.cat((history_k,k),1);all_v=torch.cat((history_v,v),1)
         y=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),all_k.transpose(1,2),
             all_v.transpose(1,2),dropout_p=0.,is_causal=False).transpose(1,2)
     x=residual(x,att.o(y.flatten(2)),e[2])
     # No cross-attention mutable cache: recomputation is pure.
     x=x+block.cross_attn(block.norm3(x),context,None,crossattn_cache=None)
     ff=block.ffn(modulate(block.norm2(x),e[4],e[3]))
     x=residual(x,ff,e[5])
     return x,k,v,aux_att,aux_cal,diagnostic,x.new_full((),float(aux_count),dtype=torch.float32),records

def configure(net,rms_mode='native_stats',qkv=False):
 net.step_graphs.clear();net.step_pool=torch.cuda.graph_pool_handle();gc.collect();torch.cuda.empty_cache()
 net.__class__=FusedNetwork;net.fused_qkv=qkv
 if qkv and not hasattr(net,'qkv_parameters'):
  net.qkv_parameters={i:(torch.cat([b.self_attn.q.weight,b.self_attn.k.weight,b.self_attn.v.weight]),torch.cat([b.self_attn.q.bias,b.self_attn.k.bias,b.self_attn.v.bias])) for i,b in enumerate(net.model.blocks)}
 for block in net.model.blocks:
  for att in [block.self_attn,block.cross_attn]:
   for norm in [att.norm_q,att.norm_k]:
    if not hasattr(norm,'opt22_original'):norm.opt22_original=norm.forward
    if rms_mode=='off':norm.forward=norm.opt22_original
    else:
     def forward(x,norm=norm,mode=rms_mode):return rms_ordered(x,norm.weight,norm.eps) if mode=='ordered' else rms(x,norm.weight,norm.eps,mode)
     norm.forward=forward
 net.rms_mode=rms_mode
 return net
