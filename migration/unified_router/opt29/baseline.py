"""Frozen original Q32K64 and Opt28 Q64K128, same 96-event validation."""
import argparse,json,torch,gc
from .setup import ROOT,PARENT
from .data import Events
from ..opt20.model import TileRouter
from ..opt16.pipeline import GroupedRouter
from ..opt27.pipeline import Pipeline as Baseline
from ..opt27.prep_probe import LargeWorkspace
from ..opt28.pipeline import Pipeline as Tile64
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((ROOT/'experiment.json').read_text());events=Events(m,a.layer)
 model=TileRouter(m,a.layer,'Tile20').cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(PARENT/f'training/layer{a.layer:02d}/Tile20/best.pt',map_location='cuda',weights_only=True)['model']);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False);rows=[]
 for item in events.val:
  q,raw,truth,targets=events.gpu(item);old=Baseline(model,raw,True,LargeWorkspace(raw));y,aux=old.run(q,raw['kc'],raw['vc'],True);row=dict(sample_id=item['row']['sample_id'],q32_nmse=float((y[:,targets].float()-truth).square().sum()/truth.square().sum()))
  new=Tile64(model,raw,aux,64,128,16);y=new.run(q,raw['kc'],raw['vc']);row['q64_nmse']=float((y[:,targets].float()-truth).square().sum()/truth.square().sum());rows.append(row);del old,new,aux,y
 result=dict(completed=True,layer=a.layer,rows=rows,q32_nmse=sum(x['q32_nmse']for x in rows)/len(rows),q64_nmse=sum(x['q64_nmse']for x in rows)/len(rows),frozen_source=str(PARENT));atomic_json(ROOT/f'baseline/layer{a.layer:02d}.json',result);print(result['q32_nmse'],result['q64_nmse'],flush=True)
if __name__=='__main__':main()
