"""Include route, regrouping, current KV work and cached native attention."""
import argparse,torch,time,importlib.util
from .setup import ROOT,tag
from .probe import load,measure
from .sparge import Sparge
from .pipeline import _sync
from .retile import select
from ..opt23.route import scores
from ..opt16.grouping import order,gather
from ..opt5.probe import capture,timing
from ..common import atomic_json
import triton as tr
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',480),('width',832),('chunks',4)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();name=tag(a.height,a.width,a.chunks)
 while not (ROOT/'confirm'/name/'layer24.json').exists():time.sleep(5)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,24);kv=aux['kv'];h,n,d=raw['q'].shape;seg=pipe.seg;b=kv.state['key'].shape[1]
 path=ROOT/'sparge_warp32_build/opt24_sparge_warp32_sm80.so';spec=importlib.util.spec_from_file_location('opt24_sparge_warp32_sm80',path);warp=importlib.util.module_from_spec(spec);spec.loader.exec_module(warp)
 base=lambda:pipe.run(raw['q'],raw['kc'],raw['vc']);bg,bo=capture(base);rows=[]
 for group in [32,128]:
  lib=Sparge(kv,seg,group);lib.ks=pipe.quant.scale
  if group==128:lib.module=warp
  def run():
   kv=pipe.cache.current(raw['kc'],raw['vc']);pipe.quant.current();start=pipe.quant.nhpages
   _sync[(b-start,h)](pipe.quant.k,kv.v,lib.ki,lib.v,kv.k.shape[1],b,start,seg,tr.cdiv(seg,64),num_warps=4)
   ix=order(raw['q'],model.tree('q'));q=gather(raw['q'],ix)
   f,c=scores(q,kv.state,.35,8,seg);ids=select(f,c,b,group)
   return lib.prepared(q,ids,ix)
  timing(bg,200 if n<10000 else 10);r,y=measure(run,bg,ref,reps=20);r.update(group=group,speedup=r['baseline_ms']/r['ms']);rows.append(r);print(r,flush=True)
 atomic_json(ROOT/f'module_{name}.json',dict(completed=True,layer=24,scope='Complete warm sparse module including current KV updates, query grouping, route, retile and native attention. History preparation excluded.',rows=rows))
if __name__=='__main__':main()
