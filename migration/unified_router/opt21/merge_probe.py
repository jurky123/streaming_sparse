import json,statistics,time
import torch
from .setup import ROOT,PARENT
from .attention import attention
from ..common import config,atomic_json,paired_seed
from ..opt17.data import Events
from ..opt20.model import TileRouter
from ..opt16.pipeline import GroupedRouter
from ..opt19.pipeline import Pipeline
from ..opt19.attention import attention as baseline
from ..opt16.grouping import restore
from ..opt5.probe import capture,timing

@torch.no_grad()
def main():
 torch.set_num_threads(4);paired_seed(0);torch.backends.cuda.matmul.allow_tf32=False
 _,m=config(PARENT);ev=Events(m,24);model=TileRouter(m,24,'Tile20').cuda().eval()
 model.load_state_dict(torch.load(PARENT/'training/layer24/Tile20/best.pt',map_location='cuda',weights_only=True)['model']);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
 rows=[]
 for nh in [4680,9360,14040]:
  event=next(x for x in ev.val if x['row']['history_tokens']==nh and x['row']['step']==3 and x['row']['generation_seed']==0)
  q,raw,_,_=ev.gpu(event);pipe=Pipeline(model,raw,32,.2,'shared','fp16');_,d=pipe.run(q,raw['kc'],raw['vc'],True)
  ix=d['q_order'];qq=q.gather(1,ix.long()[...,None].expand_as(q));kv=d['kv'];ids=d['selection']
  base=lambda:restore(baseline(qq,kv,ids,32,pipe.quant,2,2,2),ix)
  expected=base();g,_=capture(base);btime=timing(g,100);print('baseline',nh,btime,flush=True)
  for splits in [2]:
   for warps in [2]:
    for stages in [2]:
     try:
      fn=lambda:attention(qq,kv,ids,pipe.quant,ix,splits,warps,stages)
      actual=fn();err=float((actual.float()-expected.float()).square().sum()/expected.float().square().sum())
      graph,_=capture(fn)
      pairs=[]
      for rep in range(3):
       if rep%2:a=timing(graph,60);b=timing(g,60)
       else:b=timing(g,60);a=timing(graph,60)
       pairs.append([b,a])
      row=dict(history=nh,splits=splits,warps=warps,stages=stages,nmse_vs_parent=err,bit_identical=torch.equal(actual,expected),baseline_ms=statistics.median(v[0] for v in pairs),ms=statistics.median(v[1] for v in pairs));rows.append(row);print(row,flush=True)
      del graph,actual
     except Exception as e:print('failed',nh,splits,warps,stages,repr(e),flush=True)
     atomic_json(ROOT/'merge_exact_probe.json',rows)
 atomic_json(ROOT/'merge_exact_probe_completed.json',dict(completed=True,rows=len(rows)))
if __name__=='__main__':main()
