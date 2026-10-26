import argparse,math,json,traceback
import torch,triton as tr
from .setup import ROOT,tag
from .probe import measure
from .route import coarse
from .radix import top
from ..opt24.probe import load
from ..opt8.projection import project
from ..opt17.route import _top
from ..opt5.probe import capture
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',720),('width',1280),('chunks',4)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,24);q=project(aux['q_packed'],aux['kv'].state['basis']);s=coarse(q,aux['kv'].state,8,128);h,g,b=s.shape;c=math.ceil(b*.35)
 def old():
  o=torch.empty((h,g,c),device=s.device,dtype=torch.int32);_top[(h*g,)](s,o,h*g,b,c,tr.next_power_of_2(b),num_warps=4);return o
 bg,bo=capture(old);bg.replay();reference=bo.clone();rows=[]
 for bits in [4,8]:
  for warps in [4,8]:
   try:r=measure(lambda:top(s,c,bits,warps),bg,reference);r.update(bits=bits,warps=warps)
   except Exception as e:traceback.print_exc();r=dict(bits=bits,warps=warps,error=str(e))
   rows.append(r);print(r,flush=True)
 atomic_json(ROOT/f'radix_{tag(a.height,a.width,a.chunks)}.json',dict(completed=True,rows=rows))
if __name__=='__main__':main()
