"""Latest algorithm with fused original summary formulas; numerical candidate."""
import torch
from .rope_runtime import RopeLatestRuntime,RopeTailLatestRuntime
from .split_summary import build
from .ops import SummaryOps
class SummaryLatestRuntime(RopeLatestRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.fused_summary_calls=0;self.fused_summary_shapes=set()
 def build_summary(self,model,k,v,cheap=False):
  assert not torch.is_grad_enabled()
  out=build(model,k,v,cheap,three_term_features=True);self.fused_summary_calls+=1;self.fused_summary_shapes.add((cheap,tuple(k.shape),k.stride(),v.stride()))
  l,c,s=self.key();key=(l,c,s,cheap)
  if self.validate and l in (1,15,29) and c in (2,5,8,20,40) and s in (0,3,4) and key not in self.summary_checked:
   ref=SummaryOps.build(model,k,v,cheap);errors={}
   for name in ('keys','mean','detail','stats'):
    a=getattr(out,name).float();b=getattr(ref,name).float();d=a-b
    errors[name]=dict(nmse=float(d.square().mean()/b.square().mean().clamp_min(1e-12)),max_abs=float(d.abs().max()),equal_fraction=float((a==b).float().mean()),finite=bool(a.isfinite().all()))
   row=dict(layer=l,chunk=c,step=s,cheap=cheap,errors=errors)
   assert all(e['finite'] and e['nmse']<=2e-7 for e in errors.values()),row
   self.summary_validation.append(row);self.summary_checked.add(key)
  return out
 def finish(self):
  result=super().finish();result.update(implementation='SummaryLatestRuntime',summary='Four Triton kernels implementing latest pooling, full mean, Haar detail and four-risk statistics; three-term BF16 weight projection with FP32 accumulation and projected-space centering; changed floating-point reduction order',fused_summary_calls=self.fused_summary_calls,fused_summary_shapes=len(self.fused_summary_shapes),summary_graph_calls=0,summary_graph_shapes=0);return result
class SummaryTailLatestRuntime(SummaryLatestRuntime):
 def selected_attention(self,*args):return RopeTailLatestRuntime.selected_attention(self,*args)
 def finish(self):
  result=super().finish();result.update(implementation='SummaryTailLatestRuntime',attention='Existing two-term BF16 faster-tail candidate');return result
