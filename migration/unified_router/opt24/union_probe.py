import argparse,torch,traceback
from .setup import ROOT,tag
from .probe import load,measure
from .union import join,attention
from .attention import attention as original
from ..opt5.probe import capture
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',480),('width',832),('chunks',4)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;name=tag(a.height,a.width,a.chunks)
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,24);q=aux['q_packed'];kv=aux['kv'];ix=aux['q_order'];seg=pipe.seg;h,n,d=q.shape;b=kv.state['key'].shape[1]
 base=lambda:original(q,kv,aux['selection'],pipe.quant,seg,ix);bg,bo=capture(base);rows=[]
 for group in [64,128]:
  joined=join(aux['selection'],b,n,group);executed=float(joined[2].sum())*group*64/(h*n*kv.k.shape[1])
  for warps,stages,int8 in [(2,2,True),(4,2,True),(4,3,True),(4,2,False)]:
   opts=dict(group=group,warps=warps,stages=stages,int8=int8);print('compile',opts,flush=True)
   try:
    fn=lambda:attention(q,kv,joined,pipe.quant,seg,ix,**opts);r,y=measure(fn,bg,ref,reps=12);assert r['nmse_vs_parent']<(1e-4 if int8 else .005),r
    rp,_=measure(lambda:attention(q,kv,join(aux['selection'],b,n,group),pipe.quant,seg,ix,**opts),bg,ref,reps=12)
    r.update(options=opts,execution=executed,with_join_ms=rp['ms'],speedup=r['baseline_ms']/r['ms'])
   except Exception as e:traceback.print_exc();r=dict(options=opts,error=str(e))
   rows.append(r);print(r,flush=True);atomic_json(ROOT/f'union_{name}.json',dict(completed=False,rows=rows))
 atomic_json(ROOT/f'union_{name}.json',dict(completed=True,rows=rows))
if __name__=='__main__':main()
