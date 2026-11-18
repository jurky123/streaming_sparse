"""Arithmetic and isolated latency checks, including original CTA layout control."""
import json,statistics
import torch
from .group_softmax import softmax,library
from .weights import ROOT
from ..review_v1.model import Layer

@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;library()
 out=ROOT/'artifacts/video_runtime_v1/group_softmax_v1';out.mkdir(exist_ok=True)
 saved=torch.load(ROOT/'artifacts/video_runtime_v1/grouping_dataflow_v1/input.pt',weights_only=True)
 m=Layer({n:x.cuda() for n,x in saved['model'].items()}).eval().requires_grad_(False);k=saved['k'].cuda()[0].transpose(0,1).float()[:,None]
 score=torch.einsum('hptd,hrd->hptr',k-k.mean(-2,keepdim=True),m.bank)/(128**.5)
 torch.manual_seed(702);inputs={'real':score,'random':torch.randn_like(score),'large':torch.randn_like(score)*100,'constant':torch.ones_like(score),'small':torch.randn_like(score)*.001}
 contracts=[]
 for name,x in inputs.items():
  ref=x.softmax(-2)
  for columns,coalesced in [(32,False),(4,False),(4,True)]:
   y=softmax(x,columns,coalesced);row=dict(input=name,columns=columns,coalesced=coalesced,equal=torch.equal(y,ref),different=int((y!=ref).sum()),max_abs=float((y-ref).abs().max()));contracts.append(row)
 def capture(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):output=fn()
  return g,output
 graphs={name:capture(fn) for name,fn in [('torch',lambda:score.softmax(-2)),('columns32',lambda:softmax(score,32,False)),('columns4',lambda:softmax(score,4,False)),('coalesced4',lambda:softmax(score,4,True))]}
 rows={name:[] for name in graphs}
 for rep in range(15):
  order=list(graphs) if rep%2==0 else list(reversed(graphs))
  for name in order:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(20):graphs[name][0].replay()
   b.record();b.synchronize();rows[name].append(a.elapsed_time(b)/20)
 medians={name:statistics.median(values) for name,values in rows.items()}
 result=dict(contracts=contracts,medians_ms=medians,speedup=medians['torch']/medians['coalesced4'],rows=rows)
 (out/'isolated.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
