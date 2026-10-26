import argparse,torch,statistics,math,gc
from .setup import ROOT,tag
from .pipeline import Pipeline,query
from .retile import select
from .probe import validate
from ..opt27.pipeline import Pipeline as Baseline
from ..opt27.prep_probe import LargeWorkspace
from ..opt27.prep import query as oldquery
from ..opt24.probe import load
from ..opt24.sparge import quant
from ..opt16.grouping import order,gather
from ..opt5.probe import capture,timing
from ..opt23.bench import nmse
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for key,v in [('height',720),('width',1280),('chunks',4),('layer',24)]:p.add_argument('--'+key,type=int,default=v)
 p.add_argument('--adapter-v2',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 adapter_select=select
 if a.adapter_v2:
  from .retile_v2 import select as adapter_select
 raw,model,_,ref,aux=load(a.height,a.width,a.chunks,a.layer);old=Baseline(model,raw,True,LargeWorkspace(raw));fn=lambda:old.run(raw['q'],raw['kc'],raw['vc']);bg,bo=capture(fn);bg.replay();bo=bo.clone()
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
  dg,truth=capture(lambda:torch.nn.functional.scaled_dot_product_attention(raw['q'][None],aux['kv'].k[None],aux['kv'].v[None])[0]);dg.replay();truth=truth.clone()
 oldproj=oldquery(raw['q'],aux['q_order'],aux['kv'].state['basis'])[0];rows=[]
 path=ROOT/('module_v2' if a.adapter_v2 else 'module')/tag(a.height,a.width,a.chunks)/f'layer{a.layer:02d}.json';path.parent.mkdir(parents=True,exist_ok=True)
 for g,k,w in [(64,64,16 if a.height==480 else 32),(128,64,32),(64,128,16)]:
  new=Pipeline(model,raw,aux,g,k,w,select_fn=adapter_select);out,details=new.run(raw['q'],raw['kc'],raw['vc'],True);lib=new.lib;ids=details['selection']
  assert bool(((ids>=0)&(ids<lib.b)).all()) and bool((ids[:,:,1:]>ids[:,:,:-1]).all())
  qi,qs=quant(aux['q_packed'],g);checks=dict(projected_exact=torch.equal(oldproj,details['projected']),query_quant_exact=torch.equal(qi,details['qi']),query_scale_exact=torch.equal(qs,details['qs']))
  if a.adapter_v2:
   new.select=select;original_out=new.run(raw['q'],raw['kc'],raw['vc']);checks['adapter_v1_output_exact']=torch.equal(original_out,out);new.select=adapter_select;del original_out
  assert all(checks.values()),checks
  err=validate(lib,aux['q_packed'],ids,details['q_order'],out)
  cg,co=capture(lambda:new.run(raw['q'],raw['kc'],raw['vc']));cg.replay();assert torch.equal(co,out)
  timing(cg,max(5,math.ceil(350/max(timing(cg,3),.01))));samples=[]
  for rep in range(5):
   if rep%2:base=timing(bg,15);dense=timing(dg,10);ms=timing(cg,15)
   else:ms=timing(cg,15);dense=timing(dg,10);base=timing(bg,15)
   samples.append([base,ms,dense])
  components={}
  funcs={'current_kv':lambda:new.current(raw['kc'],raw['vc']),
   'query_group_and_prep':lambda:query(raw['q'],order(raw['q'],model.tree('q')),details['kv'].state['basis'],g),
   'coarse_and_fine':lambda:new.scores(details['projected'],details['kv'].state),
   'selection_adapter':lambda:adapter_select(details['fine'],details['candidates'],details['kv'].state['key'].shape[1],new.seg,g,k),
   'attention_metadata_restore':lambda:new.attention(details['qi'],details['qs'],ids,details['q_order'],raw['q'].dtype)}
  for label,f in funcs.items():
   pg,_=capture(f);timing(pg,10);components[label]=statistics.median(timing(pg,20) for _ in range(3));del pg
  base=statistics.median(x[0]for x in samples);ms=statistics.median(x[1]for x in samples);dense=statistics.median(x[2]for x in samples)
  row=dict(options=dict(q=g,k=k,warp_q=w,warps=g//w),ms=ms,baseline_ms=base,dense_ms=dense,speedup=base/ms,time_vs_dense=ms/dense,samples=samples,components_ms=components,checks=checks,reference_max_nmse=err,nmse_vs_dense=nmse(co,truth),nmse_vs_q32=nmse(co,bo),baseline_nmse_vs_dense=nmse(bo,truth))
  rows.append(row);print(row,flush=True);atomic_json(path,dict(completed=False,scope='Warm full sparse module including current KV, query grouping, route, adapter, attention and restore. History preparation excluded. Dense comparison is cached full KV FlashAttention only.',rows=rows))
  del cg,co,new,lib,details,funcs,out;gc.collect();torch.cuda.empty_cache()
 atomic_json(path,dict(completed=True,scope='Warm full sparse module including current KV, query grouping, route, adapter, attention and restore. History preparation excluded. Dense comparison is cached full KV FlashAttention only.',rows=rows))
if __name__=='__main__':main()
