"""Reuse bounded raw-KV scratch; graph replay unchanged grouping and selection."""
import torch
from .optimized_runtime import StableLatestRuntime,OptimizedLatestRuntime
from .cached_ops import GraphGrouping,GraphSelection
from .ops import RouteOps
from .runtime import GroupingOps
from .fast_attention import attention as stable_attention
from .tuned_attention import attention as fast_tail_attention
class CachedLatestRuntime(StableLatestRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.group_graph=GraphGrouping(self.config);self.select_graph=GraphSelection(self.config)
  self.cache_checks=[];self.group_checks=[];self.selection_checks=[];self.kv_reuse_calls=0
 def history_and_current(self,cache,k,v,p):
  assert not torch.is_grad_enabled() and self.config.current_pages==72
  assert 0<=p<=self.config.history_pages and k.shape[1]==4608
  check=self.should_check() or (self.validate and self.key()[0] in (1,15,29) and self.key()[1] in (2,5,8) and self.key()[2]==4)
  if check:ref=super().history_and_current(cache,k,v,p)
  end=p*64+4608
  # The region [p,p+72) is outside the committed history [0,p).
  # It is scratch only; grouped clean commit still owns every history update.
  cache['k'][:,p*64:end].copy_(k);cache['v'][:,p*64:end].copy_(v)
  out=(cache['k'][:,:end],cache['v'][:,:end]);self.kv_reuse_calls+=1
  if check:
   equal=all(torch.equal(a,b) for a,b in zip(out,ref));assert equal
   self.cache_checks.append(dict(key=self.key(),history_pages=p,bitwise_equal=equal))
  return out
 def group(self,l,k,v):
  out=self.group_graph(self.model[l],k,v)
  if self.validate and l in (1,15,29) and self.key()[1] in (2,5,8):
   ref=GroupingOps.group(self.model[l],k,v,self.config);equal=[torch.equal(a,b) for a,b in zip(out,ref)]
   assert all(equal),(self.key(),equal);self.group_checks.append(dict(key=self.key(),bitwise_equal=equal))
  return out
 def select_pages(self,scores,own,threshold):
  assert threshold is None,'Only the validated fixed-density configuration is supported'
  out=self.select_graph(scores)
  if self.should_check():
   ref=RouteOps.select(scores,own,self.config);equal={n:torch.equal(out[n],ref[n]) for n in ref}
   assert all(equal.values()),(self.key(),equal);self.selection_checks.append(dict(key=self.key(),bitwise_equal=equal))
  return out
 def selected_attention(self,*args):
  out=stable_attention(*args,exact_stages=2);self.fast_calls+=1
  if self.should_check() and self.key() not in self.attn_checked:
   ref=super().selected_attention(*args);assert torch.equal(out,ref),('Exact launch changed output',self.key())
   self.validation[-1]['exact_launch_bitwise_equal']=True
  return out
 def finish(self):
  result=super().finish();pools={}
  for name,obj in [('group',self.group_graph),('selection',self.select_graph)]:
   seg=[s for s in torch.cuda.memory_snapshot() if tuple(s.get('segment_pool_id',()))==obj.pool]
   pools[name]=sum(s['total_size'] for s in seg) if seg else None
  result.update(status='latest_algorithm_cached_execution',implementation='CachedLatestRuntime',exact_kernel_stages=2,kv_reuse_calls=self.kv_reuse_calls,kv_cache_policy='Current KV written only to existing scratch after committed history; no repeated history concatenation',group_graph_calls=self.group_graph.calls,group_graph_shapes=len(self.group_graph.entries),selection_graph_calls=self.select_graph.calls,selection_graph_shapes=len(self.select_graph.entries),extra_graph_pool_reserved_bytes=pools,cache_validation=self.cache_checks,group_validation=self.group_checks,selection_validation=self.selection_checks,workspace_accounting='Peak memory includes all graph pools; persistent_bytes includes base summary static buffers but excludes additional group/selection workspaces reported separately')
  return result
class CachedTailLatestRuntime(CachedLatestRuntime):
 """Same execution optimization with preceding faster-tail candidate unchanged."""
 def selected_attention(self,*args):
  out=fast_tail_attention(*args,bf16_av=True,exact_stages=2);self.fast_calls+=1
  if self.should_check() and self.key() not in self.attn_checked:
   ref=OptimizedLatestRuntime.selected_attention(self,*args);assert torch.equal(out,ref)
   self.validation[-1]['exact_launch_bitwise_equal']=True
  return out
 def finish(self):
  result=super().finish();result.update(implementation='CachedTailLatestRuntime',attention='Unchanged preceding optimized two-term BF16 tail candidate');return result
