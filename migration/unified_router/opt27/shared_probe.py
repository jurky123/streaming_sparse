import argparse,torch,triton as tr,json,traceback
from .setup import ROOT,tag
from .shared import refine,module
from ..opt24.probe import load
from ..opt26.probe import measure
from ..opt8.projection import project
from ..opt23.route import _fine_age
from ..opt19.tile import select
from ..opt26.route import select as fast_select
from ..opt5.probe import capture
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',720),('width',1280),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 p.add_argument('--mass',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];state=aux['kv'].state;qq=project(q,state['basis']);h,n,_=q.shape;_,g,c=aux['candidates'].shape;b=state['key'].shape[1];ids=aux['candidates']
 def old():
  f=torch.empty((h,n,c),device=q.device);_fine_age[(g,tr.cdiv(c,8),h)](qq,state['key'],state['logcount'],ids,f,n,b,c,g,tr.cdiv(n,64),num_warps=4,num_stages=1)
  return fast_select(f,ids,n,b,radix=True) if c>512 else select(f,ids,b,32,.2)
 bg,bo=capture(old);bg.replay();reference=bo.clone();mask=torch.zeros((h,g,b),device=q.device,dtype=torch.bool).scatter_(-1,reference.long(),True)
 module()
 opts=[dict(m=m,gp=gp,warps=w) for m in [32,16] for gp in [8,16] for w in [4,8]]
 out=ROOT/'shared'/tag(a.height,a.width,a.chunks)/f'layer{a.layer:02d}.json';out.parent.mkdir(parents=True,exist_ok=True);rows=[]
 for opt in opts:
  print('compile',opt,flush=True)
  try:
   fn=lambda:refine(qq,state,ids,n,**opt);r=measure(fn,bg,reference,reps=20);y=fn();assert bool(((y>=0)&(y<b)).all());ym=torch.zeros_like(mask).scatter_(-1,y.long(),True)
   r.update(options=opt,replaced_fraction=float((ym&~mask).sum()/reference.numel()),changed_tile_fraction=float((y!=reference).any(-1).float().mean()))
  except Exception as e:traceback.print_exc();r=dict(options=opt,error=str(e))
  rows.append(r);print(r,flush=True);atomic_json(out,dict(completed=False,rows=rows,original_score_bytes=h*n*c*4))
 atomic_json(out,dict(completed=True,rows=rows,original_score_bytes=h*n*c*4))
if __name__=='__main__':main()
