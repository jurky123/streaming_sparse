import argparse,torch,traceback
from .setup import ROOT,tag
from .probe import load,measure
from .attention import attention
from .retile import select
from .sparge import Sparge
from ..opt23.bench import nmse
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',480),('width',832),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;name=tag(a.height,a.width,a.chunks)
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];kv=aux['kv'];ix=aux['q_order'];b=kv.state['key'].shape[1];rows=[]
 base=lambda:attention(q,kv,aux['selection'],pipe.quant,pipe.seg,ix);bg,bo=capture(base)
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):truth=torch.nn.functional.scaled_dot_product_attention(raw['q'][None],kv.k[None],kv.v[None])[0]
 for group in [32,64,128]:
  print('compile',group,flush=True)
  try:
   ids=select(aux['fine'],aux['candidates'],b,group);lib=Sparge(kv,pipe.seg,group);expected=attention(q,kv,ids,pipe.quant,pipe.seg,ix,group=group,int8=True,warps=4)
   y=lib.prepared(q,ids,ix);err=nmse(y,expected);assert torch.isfinite(y).all() and err<.001,(group,err)
   timing(bg,300 if q.shape[1]<10000 else 15)
   r,y=measure(lambda:lib.prepared(q,ids,ix),bg,ref,reps=20)
   rp,_=measure(lambda:lib.run(q,kv,ids,ix),bg,ref,reps=20)
   r.update(group=group,nmse_vs_same_mask=err,nmse_vs_dense=nmse(y,truth),baseline_nmse_vs_dense=nmse(ref,truth),with_pack_quant_ms=rp['ms'],speedup=r['baseline_ms']/r['ms'])
  except Exception as e:traceback.print_exc();r=dict(group=group,error=str(e));rows.append(r);break
  rows.append(r);print(r,flush=True);atomic_json(ROOT/f'sparge_{name}_layer{a.layer:02d}.json',dict(completed=False,rows=rows))
 atomic_json(ROOT/f'sparge_{name}_layer{a.layer:02d}.json',dict(completed=True,rows=rows))
if __name__=='__main__':main()
