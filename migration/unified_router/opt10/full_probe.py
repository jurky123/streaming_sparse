import argparse,statistics,gc
from pathlib import Path
import torch
from ..opt9.network import PackedRoute
from .head_route import HeadRoute,HeadMixin
from .diag_route import DiagRoute
from ..opt4.network import CompactGraph
from ..learning import ActionHead
from ..opt5.probe import timing
from ..common import atomic_json
class HeadDiagRoute(HeadMixin,DiagRoute):pass
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=d['q'];head=ActionHead(d['head']).cuda();head.load_state_dict(d['head']);head.eval()
 def build(cls,batch):
  route=cls(batch=batch);g=CompactGraph(route,q,d['sketch'],d['prepared'],head,d['beta'],'K32-Set',batch,torch.cuda.graph_pool_handle());g.step.fill_(3.);g.graph.replay();return g
 ref=build(PackedRoute,512);torch.cuda.synchronize();expected=ref.output.clone();timing(ref.graph,150);rows=[]
 for name,cls in [('Wide',PackedRoute),('WideHead',HeadRoute),('WideDiag',HeadDiagRoute)]:
  new=build(cls,4680);torch.cuda.synchronize();mask=new.output.clone();changed=int((mask!=expected).any(-1).sum());timing(new.graph,150);pairs=[]
  for i in range(6):
   if i%2:nt=timing(new.graph,60);rt=timing(ref.graph,60)
   else:rt=timing(ref.graph,60);nt=timing(new.graph,60)
   pairs.append((rt,nt))
  from ..opt7.split_attention import attention
  yy=attention(q,d['k'],d['v'],mask);yr=attention(q,d['k'],d['v'],expected)
  row=dict(variant=name,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),changed_queries=changed,queries=mask.shape[0]*mask.shape[1],attention_nmse=float((yy.float()-yr.float()).square().sum()/yr.float().square().sum()),samples=pairs)
  rows.append(row);atomic_json(root/'optimization10/full_probe.json',rows);print(row,flush=True);del new;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
