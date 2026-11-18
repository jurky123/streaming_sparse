"""Bounded exact-attention accumulator/lifetime prototypes, real LUT input."""
import json,statistics,argparse
import torch
from .weights import ROOT
from ..gap_exact_round_kernel import _attn_fwd
from .exact_fold import exact_fold
from .exact_vlate import exact_vlate
from .exact_fold_vlate import exact_fold_vlate
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--pages',type=int,required=True);a=parser.parse_args();p=a.pages;count=p//10
 torch.set_num_threads(4);s=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
 q,k,v=[s[n].cuda().contiguous() for n in ('q','k','v')];k=k[:,:p*64];v=v[:,:p*64]
 raw=s['ids'].reshape(-1,43).tolist();selected=[]
 for row in raw:
  valid=[i for i in row if i<p];valid.extend(i for i in range(p) if i not in valid);selected.append(valid[:count])
 ids=torch.tensor(selected,device='cuda',dtype=torch.int32).reshape(1,12,72,count)
 root=ROOT/'artifacts/video_runtime_v1/exact_pipeline_v1';graphs=[];outputs=[];rows=[]
 for name,fn in [('baseline',_attn_fwd),('fold',exact_fold)]:
  out=torch.empty(q.shape,device='cuda',dtype=torch.float32);lse=torch.empty(12,4608,device='cuda')
  def call():return fn[(72,12)](q,k,v,128**-.5,count,ids,lse,out,12,4608,k.shape[1],72,128,64,64,HIGH_PRECISION_AV=True,num_warps=4,num_stages=2)
  compiled=call()
  for _ in range(4):call()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):call()
  graph.replay();torch.cuda.synchronize();graphs.append(graph);outputs.append((out,lse))
  ref=outputs[0];delta=out-ref[0]
  rows.append(dict(name=name,registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared,output_bitwise=torch.equal(out,ref[0]),lse_bitwise=torch.equal(lse,ref[1]),nmse=(delta.square().sum()/ref[0].square().sum()).item(),max_abs=delta.abs().max().item()))
  (root/f'{name}_p{p}.ttgir').write_text(compiled.asm['ttgir']);(root/f'{name}_p{p}.ptx').write_text(compiled.asm['ptx'])
 for graph in graphs:
  for _ in range(100):graph.replay()
 torch.cuda.synchronize();times=[[] for _ in graphs]
 for r in range(30):
  for i in (range(2) if r%2==0 else range(1,-1,-1)):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(10):graphs[i].replay()
   b.record();b.synchronize();times[i].append(a.elapsed_time(b)/10)
 med=list(map(statistics.median,times))
 for row,t,tm in zip(rows,med,times):row.update(ms=t,speedup=med[0]/t,samples_ms=tm)
 (root/f'pages{p}.json').write_text(json.dumps(dict(rows=rows,pages=p,count=count),indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='samples_ms'} for r in rows],indent=2))
if __name__=='__main__':main()
