"""Latest algorithm with fused four-risk routing and repository exact attention."""
import torch
from .fast_runtime import FastLatestRuntime
from .ablation import AblationRuntime
from .ops import RouteOps
from .fused_route import FusedRiskRoute
class FusedLatestRuntime(FastLatestRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.graph_route=FusedRiskRoute(self.config.probes);self.route_errors={}
 def route_scores(self,probes,summary,model):
  score=self.graph_route(probes,summary,model.theta)
  if self.should_check() and self.key() not in self.route_checked:
   ref=AblationRuntime.route_scores(self,probes,summary,model)
   delta=score-ref
   nmse=float(delta.square().mean()/ref.square().mean().clamp_min(1e-12));rel=float(delta.abs().max()/ref.abs().max().clamp_min(1e-12))
   h,t,p=score.shape;own=(torch.arange(t,device=score.device)+p-72)[None].expand(h,-1)
   a=RouteOps.select(score,own,self.config);b=RouteOps.select(ref,own,self.config)
   overlap=float((a['mask']&b['mask']).sum()/b['mask'].sum())
   row=dict(route_bitwise_equal=torch.equal(score,ref),route_score_nmse=nmse,route_relative_max_error=rel,route_selected_overlap=overlap)
   assert torch.isfinite(score).all() and nmse<1e-8 and rel<5e-5 and overlap>=.999,(self.key(),row)
   self.route_errors[self.key()]=row;self.route_checked.add(self.key())
  return score
 def selected_attention(self,*args):
  out=super().selected_attention(*args)
  if self.should_check() and self.validation and self.key() in self.route_errors:
   self.validation[-1].update(self.route_errors[self.key()])
  return out
 def finish(self):
  result=super().finish()
  result.update(status='latest_algorithm_fused_implementation',implementation='FusedLatestRuntime',route='Three Triton kernels: BF16 QK, FP32 normalization, unchanged four-risk equations and probe mean; original selection',route_graph_calls=0,route_graph_shapes=0,route_fused_calls=self.graph_route.calls,route_fused_shapes=len(self.graph_route.entries))
  return result
