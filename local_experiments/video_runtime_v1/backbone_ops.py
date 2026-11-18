"""Identical optional backbone transforms for native Dense and sparse tiers."""
def install(engine,a):
 from .deployment import CURRENT
 patches={}
 if getattr(a,'block_pointwise',CURRENT['block_pointwise']):
  from .block_pointwise import PointwiseBlocks
  patches['pointwise_blocks']=PointwiseBlocks(engine,optimized=True)
  patches['pointwise_blocks'].residual_norm=getattr(a,'residual_norm',False)
  patches['pointwise_blocks'].fuse_norm=getattr(a,'layernorm_mod',False)
 if getattr(a,'rms_dataflow',CURRENT['rms_dataflow']):
  from .rms_dataflow import RMSDataflow
  patches['rms_dataflow']=RMSDataflow(engine,mean_kernel=getattr(a,'rms_mean',CURRENT['rms_mean']))
  if getattr(a,'rms_mean',CURRENT['rms_mean']) and getattr(a,'rms_row',CURRENT['rms_row']):
   from .rms_mean import row_fused
   patches['rms_dataflow'].function=row_fused
 if getattr(a,'native_rope',CURRENT['native_rope']):
  from .native_rope import NativeRope
  patches['native_rope']=NativeRope()
 if getattr(a,'rounded_ffn',False):
  import torch
  device=engine.wrapper.model.blocks[0].ffn[0].weight.device
  if torch.cuda.get_device_capability(device)==(8,0):
   from .ffn_rounded_gelu import RoundedFFN
   patches['rounded_ffn']=RoundedFFN(engine)
 if (getattr(a,'rms_rope',False) and 'native_rope' in patches and 'rms_dataflow' in patches
     and getattr(a,'rms_mean',True) and getattr(a,'rms_row',True)):
  from .rms_mean import NormalizedRotary
  patches['normalized_rotary']=NormalizedRotary(engine,getattr(a,'cached_projection',False))
 if getattr(a,'repeated_context',False) and not getattr(a,'block_graph',False):
  import torch
  if torch.cuda.get_device_capability(engine.cross[0]['k'].device)==(8,0):
   from .repeated_context import RepeatedContext
   patches['repeated_context']=RepeatedContext(engine)
 return patches
