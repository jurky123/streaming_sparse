"""Cached latest algorithm plus one-kernel Q/K FP64 rotary application."""
import torch
from .cached_runtime import CachedLatestRuntime,CachedTailLatestRuntime
from .fused_rope import apply
class RopeLatestRuntime(CachedLatestRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.rope_calls=0;self.rope_validation=[]
 def apply_rotary(self,q,k,grid_sizes,freqs,start_frame):
  assert not torch.is_grad_enabled()
  out=apply(q,k,grid_sizes,freqs,start_frame);self.rope_calls+=1
  l,c,s=self.key()
  if self.validate and l in (1,15,29) and c in (2,5,8) and s in (0,3,4):
   ref=super().apply_rotary(q,k,grid_sizes,freqs,start_frame);equal=[torch.equal(a,b) for a,b in zip(out,ref)]
   assert all(equal),(self.key(),equal);self.rope_validation.append(dict(key=self.key(),bitwise_equal=equal))
  return out
 def finish(self):
  result=super().finish();result.update(implementation='RopeLatestRuntime',rope='Single Q/K kernel with original FP64 arithmetic and final FP32/BF16 conversion',rope_calls=self.rope_calls,rope_validation=self.rope_validation);return result
class RopeTailLatestRuntime(RopeLatestRuntime):
 def selected_attention(self,*args):return CachedTailLatestRuntime.selected_attention(self,*args)
 def finish(self):
  result=super().finish();result.update(implementation='RopeTailLatestRuntime',attention='Unchanged preceding optimized two-term BF16 tail candidate');return result
