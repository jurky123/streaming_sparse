import argparse,statistics,gc
from pathlib import Path
import torch
from ..opt9.network import PackedRoute
from ..opt4.network import CompactGraph
from ..learning import ActionHead
from ..opt5.probe import timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);a=p.parse_args();root=Path(a.directory);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False);q=d['q'];head=ActionHead(d['head']).cuda();head.load_state_dict(d['head']);head.eval()
 def build(batch):
  route=PackedRoute(batch=batch);g=CompactGraph(route,q,d['sketch'],d['prepared'],head,d['beta'],'K32-Set',batch,torch.cuda.graph_pool_handle());g.step.fill_(3.);g.graph.replay();return g
 ref=build(512);ref.graph.replay();torch.cuda.synchronize();expected=ref.output.clone();timing(ref.graph,150);rows=[]
 for batch in [128,256,768,1024,1536,2048,4680][a.rank::a.world]:
  try:
   new=build(batch);torch.cuda.synchronize();changed=int((new.output!=expected).any(-1).sum());timing(new.graph,150);pairs=[]
   for i in range(6):
    if i%2:nt=timing(new.graph,60);rt=timing(ref.graph,60)
    else:rt=timing(ref.graph,60);nt=timing(new.graph,60)
    pairs.append((rt,nt))
   row=dict(batch=batch,reference_ms=statistics.median(p[0]for p in pairs),new_ms=statistics.median(p[1]for p in pairs),changed_queries=changed,samples=pairs);del new;gc.collect();torch.cuda.empty_cache()
  except Exception as e:row=dict(batch=batch,error=str(e))
  rows.append(row);atomic_json(root/f'optimization10/batch_probe{a.rank}.json',rows);print(row,flush=True)
if __name__=='__main__':main()
