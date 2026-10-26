"""Limited measured probe, no production switch or retraining."""
import argparse,json,statistics,torch
from .setup import ROOT,TRAINING,tag
from .pipeline import Pipeline
from .attention import attention as baseline
from .wide import attention
from .bench import nmse
from ..common import config,atomic_json
from ..opt20.model import TileRouter
from ..opt16.pipeline import GroupedRouter
from ..opt5.probe import capture,timing
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--height',type=int,default=480);p.add_argument('--width',type=int,default=832);p.add_argument('--chunks',type=int,default=4);p.add_argument('--layer',type=int,default=24);a=p.parse_args();name=tag(a.height,a.width,a.chunks);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw=torch.load(ROOT/'capture'/name/f'layer{a.layer:02d}.pt',map_location='cuda',weights_only=False);_,m=config(TRAINING)
 model=TileRouter(m,a.layer,'Tile20').cuda().eval();model.load_state_dict(torch.load(TRAINING/f'training/layer{a.layer:02d}/Tile20/best.pt',map_location='cuda',weights_only=True)['model']);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
 pipe=Pipeline(model,raw,raw['chunk_tokens']);expected,aux=pipe.run(raw['q'],raw['kc'],raw['vc'],True)
 args=(aux['q_packed'],aux['kv'],aux['selection'],pipe.quant,pipe.seg,aux['q_order']);base=lambda:baseline(*args);bg,bo=capture(base);bg.replay();assert torch.equal(bo,expected);rows=[]
 for bn,warps,stages in [(128,2,2),(128,4,2),(128,4,3),(128,4,4),(256,4,2),(256,8,2)]:
  options=dict(bn=bn,warps=warps,stages=stages);print('compile',options,flush=True)
  fn=lambda:attention(*args,**options);y=fn();assert torch.isfinite(y).all();err=nmse(y,expected);g,o=capture(fn);g.replay();assert torch.equal(o,y)
  timing(bg,10);timing(g,10);pairs=[]
  for i in range(5):
   if i%2:b=timing(bg,30);s=timing(g,30)
   else:s=timing(g,30);b=timing(bg,30)
   pairs.append([b,s])
  row=dict(options=options,nmse_vs_parent=err,baseline_ms=statistics.median(x[0] for x in pairs),ms=statistics.median(x[1] for x in pairs),samples=pairs);row['speedup']=row['baseline_ms']/row['ms'];rows.append(row);print(row,flush=True);del g,o,y
  atomic_json(ROOT/f'wide_probe_{name}_layer{a.layer:02d}.json',dict(case=name,layer=a.layer,completed=False,rows=rows))
 atomic_json(ROOT/f'wide_probe_{name}_layer{a.layer:02d}.json',dict(case=name,layer=a.layer,completed=True,rows=rows))
if __name__=='__main__':main()
