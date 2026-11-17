"""Latest algorithm: batched fused routing, graph summaries, two-term tail AV."""
import torch
from .fused_runtime import FusedLatestRuntime
from .graph_summary import GraphSummary
from .ops import SummaryOps
from .runtime import SelectedAttention
from .tuned_attention import attention
from .fast_attention import attention as previous_attention
class OptimizedLatestRuntime(FusedLatestRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.summary_graph=GraphSummary();self.summary_validation=[];self.summary_checked=set();self.batched_checked=set();self.probe_offsets=None
 def build_summary(self,model,k,v,cheap=False):
  assert not torch.is_grad_enabled(),'Optimized summary path is inference-only'
  out=self.summary_graph(model,k,v,cheap)
  l,c,s=self.key();key=(l,c,s,cheap)
  if self.validate and l in (1,15,29) and c in (2,5,8) and s in (0,3,4) and key not in self.summary_checked:
   ref=SummaryOps.build(model,k,v,cheap)
   equal={n:torch.equal(getattr(ref,n),getattr(out,n)) for n in ('keys','mean','detail','stats')}
   row=dict(layer=l,chunk=c,step=s,cheap=cheap,bitwise_equal=equal)
   assert all(equal.values()),row
   self.summary_validation.append(row);self.summary_checked.add(key)
  return out
 def query_route_scores(self,query,summary,model):
  if self.probe_offsets is None:self.probe_offsets=torch.linspace(0,63,self.config.probes,device=query.device).round().long()
  probes=query.reshape(12,72,64,128).index_select(2,self.probe_offsets).flatten(1,2)
  score=self.graph_route(probes,summary,model.theta)
  if self.should_check() and self.key() not in self.batched_checked:
   # Retain the original FP32 score/selection gate on the first eight tiles.
   super().route_scores(probes[:,:128],summary,model)
   previous=torch.cat([self.graph_route(probes[:,i:i+128],summary,model.theta) for i in range(0,1152,128)],1)
   assert torch.equal(score,previous),('Batched route differs',self.key())
   self.route_errors[self.key()]['batched_route_bitwise_equal']=True;self.batched_checked.add(self.key())
  return score
 def selected_attention(self,*args):
  out=attention(*args,bq=64,br=64,bf16_av=True);self.fast_calls+=1
  if self.should_check() and self.key() not in self.attn_checked:
   ref=SelectedAttention.run(*args);previous=previous_attention(*args)
   delta=out.float()-ref.float();diff=out.float()-previous.float()
   nmse=float(delta.square().mean()/ref.float().square().mean().clamp_min(1e-12));err=float(delta.abs().max())
   row=dict(layer=self.key()[0],chunk=self.key()[1],step=self.key()[2],nmse=nmse,max_abs=err,previous_attention_nmse=float(diff.square().mean()/previous.float().square().mean().clamp_min(1e-12)),previous_attention_max_abs=float(diff.abs().max()))
   row.update(self.route_errors[self.key()])
   assert torch.isfinite(out).all() and nmse<=2e-5 and err<=.05,row
   self.validation.append(row);self.attn_checked.add(self.key())
  return out
 def finish(self):
  result=super().finish();buffers=0
  for kk,vv,mm,graph,summary,ha in self.summary_graph.entries.values():
   tensors=[kk,vv,ha,*mm.parameters(),*(getattr(summary,n) for n in ('keys','mean','detail','stats'))]
   buffers+=sum(t.numel()*t.element_size() for t in tensors)
  segments=[s for s in torch.cuda.memory_snapshot() if tuple(s.get('segment_pool_id',()))==self.summary_graph.pool]
  result.update(status='latest_algorithm_optimized_implementation',implementation='OptimizedLatestRuntime',route='Same fused four-risk equations, 72 tiles in one invocation',attention='Repository page64 exact and latest full-mean/detail tail; two-term BF16 AV with FP32 accumulation',summary='Original SummaryOps equations in staged CUDA Graphs, preserving input strides and refreshing weights',summary_graph_calls=self.summary_graph.calls,summary_graph_shapes=len(self.summary_graph.entries),summary_graph_static_buffer_bytes=buffers,summary_graph_pool_reserved_bytes=sum(s['total_size'] for s in segments) if segments else None,summary_validation=self.summary_validation,workspace_accounting='Peak allocated memory includes graph workspaces; persistent_bytes additionally includes graph static buffers, not all graph pool temporary allocations')
  result['persistent_bytes']+=buffers;result['within_limit']=result['persistent_bytes']<=result['limit_bytes']
  return result

class StableLatestRuntime(OptimizedLatestRuntime):
 """Only graph summaries and batched routing; retain preceding attention exactly."""
 def selected_attention(self,*args):
  return FusedLatestRuntime.selected_attention(self,*args)
 def finish(self):
  result=super().finish()
  result.update(implementation='StableLatestRuntime',attention='Unchanged previous page64 exact + tf32x3 full-mean/detail tail',status='latest_algorithm_stable_execution_optimization')
  return result
