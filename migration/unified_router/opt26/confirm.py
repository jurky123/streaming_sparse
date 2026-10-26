"""Paired whole-router timing and exact selected-page validation."""
import argparse,math,torch
from .setup import ROOT,tag
from .probe import measure
from .route import route,select as fast_select
from ..opt24.probe import load
from ..opt23.route import scores
from ..opt19.tile import select
from ..opt5.probe import capture
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',720),('width',1280),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];state=aux['kv'].state;b=state['key'].shape[1];n=q.shape[1]
 def old():
  f,c=scores(q,state,.35,8,n);return select(f,c,b,32,.2)
 bg,bo=capture(old);bg.replay();reference=bo.clone();rows=[]
 for label,opts in [('coarse_only',dict(mg=8,width=64)),('coarse_fine2',dict(mg=8,width=64,fine_warps=2)),('coarse_radix',dict(mg=8,width=64,radix=True)),('coarse_fine2_radix',dict(mg=8,width=64,fine_warps=2,radix=True)),('coarse_fine16_radix',dict(mg=8,width=128,gp=16,radix=True))]:
  r=measure(lambda:route(q,state,n,**opts),bg,reference);r.update(label=label,options=opts);rows.append(r);print(r,flush=True)
 fn=lambda:select(aux['fine'],aux['candidates'],b,32,.2);sg,so=capture(fn);sg.replay()
 tile=measure(lambda:fast_select(aux['fine'],aux['candidates'],n,b,radix=True),sg,so.clone());print('tile_radix',tile,flush=True)
 out=ROOT/'confirm'/tag(a.height,a.width,a.chunks)/f'layer{a.layer:02d}.json';out.parent.mkdir(parents=True,exist_ok=True)
 atomic_json(out,dict(completed=True,rows=rows,tile_radix=tile))
if __name__=='__main__':main()
