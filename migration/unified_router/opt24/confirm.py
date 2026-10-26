"""Balanced final comparisons: Dense and the best native query-tile candidates."""
import argparse,torch,statistics
from types import SimpleNamespace
from .setup import ROOT,tag
from .probe import load
from .attention import attention
from .retile import select
from .sparge import Sparge
from ..opt5.probe import capture,timing
from ..opt23.bench import nmse
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',480),('width',832),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;name=tag(a.height,a.width,a.chunks);folder=ROOT/'confirm'/name;folder.mkdir(parents=True,exist_ok=True)
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];kv=aux['kv'];ix=aux['q_order'];h,n,d=q.shape;b=kv.state['key'].shape[1];kn=kv.k.shape[1]
 # Load already-built optional warp32 binding, without mutating measured sources.
 import importlib.util
 path=ROOT/'sparge_warp32_build/opt24_sparge_warp32_sm80.so';spec=importlib.util.spec_from_file_location('opt24_sparge_warp32_sm80',path);warp=importlib.util.module_from_spec(spec);spec.loader.exec_module(warp)
 functions={'Dense':lambda:torch.nn.functional.scaled_dot_product_attention(raw['q'][None],kv.k[None],kv.v[None])[0], 'Opt22_Q32':lambda:attention(q,kv,aux['selection'],pipe.quant,pipe.seg,ix)};libraries={};selections={};graphs={};outputs={}
 for g in [32,64,128]:
  ids=select(aux['fine'],aux['candidates'],b,g);lib=Sparge(kv,pipe.seg,g)
  if g==128 or (g==64 and n>10000):lib.module=warp
  key=f'Sparge_Q{g}';libraries[key]=lib;selections[key]=ids;functions[key]=lambda lib=lib,ids=ids:lib.prepared(q,ids,ix)
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
  for key,fn in functions.items():graphs[key],outputs[key]=capture(fn);graphs[key].replay()
  assert torch.equal(outputs['Opt22_Q32'],ref)
  for key in functions:
   duration=timing(graphs[key],3);timing(graphs[key],max(5,int(400/max(duration,.01))))
  samples={key:[] for key in functions}
  for rep in range(5):
   keys=list(functions);keys=keys[rep:]+keys[:rep]
   if rep%2:keys.reverse()
   for key in keys:samples[key].append(timing(graphs[key],20))
 times={key:statistics.median(v)for key,v in samples.items()};rows=[]
 for key in functions:
  row=dict(name=key,ms=times[key],samples=samples[key],speedup_vs_opt22=times['Opt22_Q32']/times[key],time_vs_dense=times[key]/times['Dense'],nmse_vs_dense=nmse(outputs[key],outputs['Dense']),nmse_vs_parent=nmse(outputs[key],ref))
  if key in selections:
   g=int(key.split('Q')[-1]);ids=selections[key];row['query_tile']=g;row['execution']=ids.numel()*g*64/(h*n*kn);row['warp_q']=32 if g==128 or(g==64 and n>10000) else 16
  rows.append(row)
 atomic_json(folder/f'layer{a.layer:02d}.json',dict(completed=True,case=name,layer=a.layer,scope='Prepared history KV; includes current Q quantization, LUT generation, attention, query restore; excludes routing and KV layout updates',rows=rows));print(rows)
if __name__=='__main__':main()
