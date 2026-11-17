"""Same captured route, Q64 vs identical M16 entrance; both correction tiers."""
import json
import torch
from .weights import ROOT
from .deployment import current,configure,load_weights
from .pipeline_graph import BatchGraphs
from .risk_projected import risk
from .profile_route import capture
from .benchmark import measure
from ..review_v1.ops import Config,Summary
from . import pipeline_graph
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;configure(current());pipeline_graph.CACHED_LOGMASS=True
 _,_,phi,_,identity=load_weights(current())
 saved=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
 e={k:v.cuda() for k,v in saved.items() if k not in ('summary','en')};e['s']=Summary(*(saved['summary'][n].cuda() for n in ('keys','mean','detail','stats')));e['en']={k:(v.cuda().bfloat16() if k in ('co','A') else v.cuda()) for k,v in saved['en'].items()}
 root=ROOT/'artifacts/video_runtime_v1/route_m16_v1';root.mkdir(parents=True,exist_ok=True)
 # Production qq is [1,4608,12,128] BF16, with head/query transpose view.
 source=e['q'].reshape(12,4608,128).transpose(0,1).contiguous().bfloat16();raw=source.transpose(0,1).reshape(12,72,64,128)
 result={}
 for full in [False,True]:
  graphs={};outs={};rows={};entries={}
  for m16 in [False,True]:
   rt=BatchGraphs(risk,phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True,input_m16=m16)
   q=(raw.index_select(2,e['offsets']) if m16 else raw).float()
   en=rt.get(q,e['s'],e['en'],e['theta'],e['basis'],e['own'],full);entries[m16]=(rt,en)
   def call(rt=rt,en=en,m16=m16):
    q=(raw.index_select(2,e['offsets']) if m16 else raw).float()
    en['q'].copy_(q);en['own'].copy_(e['own'])
    return rt.compute(en,full)
   graph,out=capture(call);graphs[m16]=graph;outs[m16]=tuple(x.clone() for x in out)
   # Outer capture makes compute inline; count conversion/copy + compute in one graph.
   rows[str(m16)]=dict(query_bytes=en['q'].numel()*en['q'].element_size())
  for _ in range(30):
   for g in graphs.values():g.replay()
  for m16,g in graphs.items():
   rows[str(m16)]['with_input']=measure(g.replay,50)
   rows[str(m16)]['core']=measure(entries[m16][1]['graph'].replay,50)
  result[str(full)]=dict(rows=rows,equal=[torch.equal(x,y) for x,y in zip(outs[False],outs[True])],max_abs=[float((x.double()-y.double()).abs().max()) for x,y in zip(outs[False],outs[True])],risk_relative_max=float((outs[False][3]-outs[True][3]).abs().max()/outs[False][3].abs().max()),risk_nmse=float((outs[False][3].double()-outs[True][3].double()).square().sum()/outs[False][3].double().square().sum()))
 (root/'result.json').write_text(json.dumps(dict(tiers=result,weights=identity),indent=2));print({k:dict(equal=v['equal'],times={n:r['with_input']['gpu_ms_median'] for n,r in v['rows'].items()}) for k,v in result.items()},flush=True)
if __name__=='__main__':main()
