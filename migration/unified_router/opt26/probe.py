import argparse,json,statistics,traceback,math
import torch,triton as tr
from .setup import ROOT,tag
from . import route as new
from ..opt24.probe import load
from ..opt8.projection import project
from ..opt17.route import _landmark,_coarse,_top
from ..opt19.tile import select
from ..opt23.route import scores
from ..opt5.probe import capture,timing
from ..common import atomic_json

def measure(fn,baseline,reference,kind='float',reps=30):
 value=fn();g,o=capture(fn);g.replay();torch.cuda.synchronize();assert torch.equal(value,o)
 diff=(value!=reference)
 row=dict(exact=not bool(diff.any()),changed=int(diff.sum()),max_abs=float((value.float()-reference.float()).abs().max()))
 timing(g,max(10,math.ceil(300/max(timing(g,5),.01))))
 samples=[]
 for i in range(5):
  if i%2:a=timing(baseline,reps);b=timing(g,reps)
  else:b=timing(g,reps);a=timing(baseline,reps)
  samples.append([a,b])
 row.update(base_ms=statistics.median(r[0] for r in samples),ms=statistics.median(r[1]for r in samples),samples=samples)
 row['speedup']=row['base_ms']/row['ms'];return row

@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',720),('width',1280),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;name=tag(a.height,a.width,a.chunks);out=ROOT/'scan'/name/f'layer{a.layer:02d}.json';out.parent.mkdir(parents=True,exist_ok=True)
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];state=aux['kv'].state;h,n,_=q.shape;g=tr.cdiv(n,32);b=state['key'].shape[1];c=aux['candidates'].shape[-1];qq=project(q,state['basis']);rows=[]
 def old_coarse():
  lm=torch.empty((h,g,8,64),device=q.device,dtype=torch.float16);_landmark[(g,h)](qq,lm,n,g,8,num_warps=4);co=torch.empty((h,g,b),device=q.device);off=0
  for kk,lc in state['parts']:
   np,p=lc.shape[1:];gp=128//p;_coarse[(g,tr.cdiv(np,gp),h)](lm,kk,lc,co,n,b,g,off,np,p,gp,8,num_warps=4,num_stages=1);off+=np
  return co
 bg,bo=capture(old_coarse);bg.replay();reference=bo.clone()
 for mg in [2,4,8,16]:
  for width in [64,128,256]:
   for warps in [4,8]:
    opts=dict(mg=mg,width=width,warps=warps)
    try:r=measure(lambda:new.coarse(qq,state,**opts),bg,reference);r.update(kind='coarse',options=opts)
    except Exception as e:traceback.print_exc();r=dict(kind='coarse',options=opts,error=str(e))
    rows.append(r);print(r,flush=True);atomic_json(out,dict(completed=False,rows=rows))
 # Fine + tile reduction together; compare final page IDs, not just scores.
 def old_fine():return select(aux['fine'],aux['candidates'],b,32,.2)
 from ..opt23.route import _fine_age
 def fine_base():
  f=torch.empty_like(aux['fine']);_fine_age[(g,tr.cdiv(c,8),h)](qq,state['key'],state['logcount'],aux['candidates'],f,n,b,c,g,tr.cdiv(n,64),num_warps=4,num_stages=1)
  return select(f,aux['candidates'],b,32,.2)
 bg,bo=capture(fine_base);bg.replay();reference=bo.clone()
 for gp in [4,8,16,32]:
  for warps in [2,4,8]:
   for trans in [False,True]:
    opts=dict(gp=gp,warps=warps,trans=trans)
    def fn():return new.select(new.refine(qq,state,aux['candidates'],n,**opts),aux['candidates'],n,b,trans)
    try:r=measure(fn,bg,reference);r.update(kind='fine',options=opts)
    except Exception as e:traceback.print_exc();r=dict(kind='fine',options=opts,error=str(e))
    rows.append(r);print(r,flush=True);atomic_json(out,dict(completed=False,rows=rows))
 atomic_json(out,dict(completed=True,rows=rows))
if __name__=='__main__':main()
