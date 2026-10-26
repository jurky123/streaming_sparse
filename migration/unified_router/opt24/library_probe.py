import argparse,json,torch,traceback,gc
from .setup import ROOT,tag
from .probe import load,measure
from .retile import select
from .attention import attention
from .library import FlashInfer,Flex,pack
from ..opt23.bench import nmse
from ..opt5.probe import capture
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--backend',choices=['flashinfer','flex'],required=True)
 for k,v in [('height',480),('width',832),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;name=tag(a.height,a.width,a.chunks);folder=ROOT/'library'/name;folder.mkdir(parents=True,exist_ok=True)
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];kv=aux['kv'];ix=aux['q_order'];h,n,d=q.shape;seg=pipe.seg;b=kv.state['key'].shape[1];pk,pv=pack(kv,seg)
 base=lambda:attention(q,kv,aux['selection'],pipe.quant,seg,ix);bg,bo=capture(base)
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):truth=torch.nn.functional.scaled_dot_product_attention(raw['q'][None],kv.k[None],kv.v[None])[0]
 rows=[]
 for group in [32,64,128,16]:
  print('compile',a.backend,group,flush=True)
  try:
   ids=select(aux['fine'],aux['candidates'],b,group);lib=(FlashInfer if a.backend=='flashinfer' else Flex)(h,n,b,group,ids.shape[-1],seg)
   expected=attention(q,kv,ids,pipe.quant,seg,ix,group=group,int8=False,warps=4)
   fn=lambda:lib.prepared(q,pk,pv,ids,ix);r,y=measure(fn,bg,ref,reps=12)
   r.update(backend=a.backend,version=lib.version,group=group,nmse_vs_same_mask=nmse(y,expected),nmse_vs_dense=nmse(y,truth),baseline_nmse_vs_dense=nmse(ref,truth));assert r['nmse_vs_same_mask']<1e-4,r
   rp,_=measure(lambda:lib.run(q,kv,ids,ix),bg,ref,reps=12);r['with_pack_ms']=rp['ms'];r['execution']=ids.numel()*group*64/(h*n*kv.k.shape[1]);r['speedup']=r['baseline_ms']/r['ms']
  except Exception as e:traceback.print_exc();r=dict(backend=a.backend,group=group,error=str(e))
  rows.append(r);print(r,flush=True);atomic_json(folder/f'{a.backend}_layer{a.layer:02d}.json',dict(completed=False,rows=rows))
 atomic_json(folder/f'{a.backend}_layer{a.layer:02d}.json',dict(completed=True,rows=rows))
if __name__=='__main__':main()
