"""Verify head gating preserves inactive heads and scores only the requested events."""
import json
import torch
from .weights import ROOT
from .deployment import current,configure,load_weights
from .pipeline_graph import BatchGraphs
from .risk_projected import risk
from ..review_v1.ops import Config,Summary
from . import pipeline_graph
@torch.no_grad()
def main():
 torch.set_num_threads(4);configure(current());pipeline_graph.CACHED_LOGMASS=True
 _,_,phi,_,_=load_weights(current())
 saved=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
 e={k:v.cuda() for k,v in saved.items() if k not in ('summary','en')};e['q']=e['q'].index_select(2,e['offsets']);e['s']=Summary(*(saved['summary'][n].cuda() for n in ('keys','mean','detail','stats')));e['en']={k:(v.cuda().bfloat16() if k in ('co','A') else v.cuda()) for k,v in saved['en'].items()}
 rt=BatchGraphs(risk,phi,Config(),compile_route=True,action_kernel=True,tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True,input_m16=True,static_route=True)
 base=rt.compute(e,False);full=rt.compute(e,True);rows=[]
 for ids in [[0],[7],[11],[0,3,6]]:
  h=torch.tensor(ids,device='cuda');en=rt.get(e['q'],e['s'],e['en'],e['theta'],e['basis'],e['own'],True,heads=h)
  rt.replay(en,e['q'],e['own']);out=tuple(x.clone() for x in en['out']);other=[i for i in range(12) if i not in ids]
  row=dict(heads=ids,inactive_mask_equal=torch.equal(out[2][other],base[2][other]),inactive_ids_equal=torch.equal(out[0][other],base[0][other]),active_mask_equal_full=torch.equal(out[2][ids],full[2][ids]),scored=int(out[4][1]),expected=len(ids)*72)
  assert row['inactive_mask_equal'] and row['inactive_ids_equal'] and row['scored']==row['expected'],row
  rows.append(row)
 root=ROOT/'artifacts/video_runtime_v1/pset_scope_v1';root.mkdir(parents=True,exist_ok=True);(root/'contracts.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
if __name__=='__main__':main()
