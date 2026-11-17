"""Latest summary_pretrained implementation with validated execution acceleration.
All grouping, summaries, route mathematics, selection and clean commits inherit
AblationRuntime unchanged. No legacy gain route or sparse checkpoint is loaded.
"""
import torch
from .ablation import AblationRuntime
from .runtime import SelectedAttention
from .fast_attention import attention
from .fast_route import GraphRoute
class FastLatestRuntime(AblationRuntime):
 def __init__(self,*args,validate=False,**kwargs):
  super().__init__(*args,**kwargs)
  assert self.variant=='no_downstream' and self.thresholds is None
  self.graph_route=GraphRoute(self.config.probes);self.validate=validate;self.validation=[];self.route_checked=set();self.attn_checked=set();self.fast_calls=0
 def key(self):
  c=self.engine.current
  return (c['layer'],c['chunk'],c['step'])
 def should_check(self):
  l,c,s=self.key();return self.validate and l in (1,15,29) and c in (2,5,8) and s in (0,3)
 def route_scores(self,probes,summary,model):
  score=self.graph_route(probes,summary,model.theta)
  if self.should_check() and self.key() not in self.route_checked:
   ref=super().route_scores(probes,summary,model)
   assert torch.equal(score,ref),('Route mismatch',self.key(),float((score-ref).abs().max()))
   self.route_checked.add(self.key())
  return score
 def selected_attention(self,*args):
  out=attention(*args);self.fast_calls+=1
  if self.should_check() and self.key() not in self.attn_checked:
   ref=SelectedAttention.run(*args);delta=out.float()-ref.float()
   nmse=float(delta.square().mean()/ref.float().square().mean().clamp_min(1e-12));err=float(delta.abs().max())
   row=dict(layer=self.key()[0],chunk=self.key()[1],step=self.key()[2],nmse=nmse,max_abs=err,route_bitwise_equal=self.key() in self.route_checked)
   assert torch.isfinite(out).all() and nmse<=2e-5 and err<=.05,row
   self.validation.append(row);self.attn_checked.add(self.key())
  return out
 def finish(self):
  result=super().finish()
  result.update(status='latest_algorithm_fast_implementation',implementation='FastLatestRuntime',algorithm='summary_pretrained/no_downstream',route='original FP32 features, unchanged selection, staged CUDA Graph replay',attention='repository page64 exact kernel with two-term AV; latest full128 mean plus zero-sum detail Triton tail',fast_attention_calls=self.fast_calls,route_graph_calls=self.graph_route.calls,route_graph_shapes=len(self.graph_route.entries),validation=self.validation)
  return result
