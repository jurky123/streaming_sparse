import argparse,statistics,json,torch,gc,traceback
from .setup import ROOT,CAPTURE,TRAINING,tag
from .retile import select
from .attention import attention
from ..opt23.pipeline import Pipeline
from ..opt23.bench import nmse
from ..opt20.model import TileRouter
from ..opt16.pipeline import GroupedRouter
from ..common import config,atomic_json
from ..opt5.probe import capture,timing

def load(h,w,c,layer):
 raw=torch.load(CAPTURE/tag(h,w,c)/f'layer{layer:02d}.pt',map_location='cuda',weights_only=False);_,m=config(TRAINING)
 model=TileRouter(m,layer,'Tile20').cuda().eval();model.load_state_dict(torch.load(TRAINING/f'training/layer{layer:02d}/Tile20/best.pt',map_location='cuda',weights_only=True)['model']);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
 pipe=Pipeline(model,raw,raw['chunk_tokens']);y,aux=pipe.run(raw['q'],raw['kc'],raw['vc'],True)
 return raw,model,pipe,y,aux

def measure(fn,base,ref,reps=20):
 y=fn();assert torch.isfinite(y).all();g,o=capture(fn);g.replay();assert torch.equal(o,y)
 timing(g,10);timing(base,10);pairs=[]
 for i in range(3):
  if i%2:b=timing(base,reps);s=timing(g,reps)
  else:s=timing(g,reps);b=timing(base,reps)
  pairs.append([b,s])
 return dict(ms=statistics.median(x[1]for x in pairs),baseline_ms=statistics.median(x[0]for x in pairs),samples=pairs,nmse_vs_parent=nmse(y,ref)),y

@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',480),('width',832),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;name=tag(a.height,a.width,a.chunks);folder=ROOT/'custom'/name;folder.mkdir(parents=True,exist_ok=True)
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];kv=aux['kv'];ix=aux['q_order'];seg=pipe.seg;h,n,d=q.shape
 base=lambda:attention(q,kv,aux['selection'],pipe.quant,seg,ix);bg,bo=capture(base);bg.replay();assert torch.equal(bo,ref)
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):truth=torch.nn.functional.scaled_dot_product_attention(raw['q'][None],kv.k[None],kv.v[None])[0]
 rows=[]
 for group in [16,32,64,128]:
  ids=select(aux['fine'],aux['candidates'],kv.state['key'].shape[1],group)
  configs=[(True,s,w,stage) for s,w,stage in [(1,2,2),(2,2,2),(1,4,2),(2,4,2),(2,4,3)]]+[(False,2,4,2)]
  for int8,splits,warps,stages in configs:
   opts=dict(group=group,int8=int8,splits=splits,warps=warps,stages=stages);print('compile',opts,flush=True)
   try:
    fn=lambda:attention(q,kv,ids,pipe.quant,seg,ix,**opts);r,y=measure(fn,bg,ref)
    r.update(options=opts,nmse_vs_dense=nmse(y,truth),baseline_nmse_vs_dense=nmse(ref,truth),execution=ids.numel()*group*64/(h*n*kv.k.shape[1]));r['speedup']=r['baseline_ms']/r['ms'];del y
   except Exception as e:traceback.print_exc();r=dict(options=opts,error=str(e))
   rows.append(r);print(r,flush=True);atomic_json(folder/f'layer{a.layer:02d}.json',dict(completed=False,rows=rows))
 atomic_json(folder/f'layer{a.layer:02d}.json',dict(completed=True,rows=rows))
if __name__=='__main__':main()
