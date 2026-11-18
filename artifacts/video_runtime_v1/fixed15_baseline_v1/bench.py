import argparse,json
from pathlib import Path
from statistics import median
import torch
from local_experiments.review_v1.tuned_attention import _tail
from paired import paired

ap=argparse.ArgumentParser()
ap.add_argument('pages',type=int);ap.add_argument('--profile',action='store_true')
ap.add_argument('--baseline-only', action='store_true')
ap.add_argument('--paired-stages',type=int,default=2)
args=ap.parse_args();p=args.pages
d=Path(__file__).parent
data=torch.load(f'/tmp/jzk_fixed15_input_{p}.pt',weights_only=True)
q,k,v,ids,basis=(data[n].cuda().contiguous() for n in ('q','k','v','ids','basis'))
basis=basis.bfloat16().contiguous();hp=data['hp']
keys,mean,detail=[torch.empty((12,hp,*x.shape[2:]),device='cuda',dtype=x.dtype)[:,:p].copy_(x) for x in (data['summary'][n] for n in ('keys','mean','detail'))]
n,h,D=q.shape[1:];count=ids.shape[-1]
old,new=torch.empty_like(q),torch.empty_like(q)
def base():
 return _tail[(72,12)](q,keys,mean,detail,None,None,None,None,p,n,h,D,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=old,HEAD_PAGES=hp,K=k,V=v,IDS=ids,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=count,QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
def candidate():
 return paired[(36,12)](q,keys,mean,detail,basis,new,k,v,ids,p,n,h,D,count,hp,num_warps=8,num_stages=args.paired_stages,enable_fp_fusion=False)
def timed(f):
 for _ in range(60):f()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):f()
 return graph
def times(graphs):
 ts=[[] for _ in graphs]
 for rep in range(20):
  for i in (list(range(len(graphs))) if rep%2==0 else list(reversed(range(len(graphs))))):
   a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
   for _ in range(8):graphs[i].replay()
   b.record();b.synchronize();ts[i].append(a.elapsed_time(b)/8)
 return ts
if __name__=='__main__':
 b=base()
 if args.baseline_only:
  samples=times([timed(base)])[0]
  row=dict(pages=p,exact=count,ms=median(samples),registers=b.n_regs,spills=b.n_spills,shared_bytes=b.metadata.shared,samples=samples)
  (d/f'baseline_{p}.json').write_text(json.dumps(row,indent=2))
 elif args.profile:
  for _ in range(50):base()
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();base();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 else:
  a=candidate()
  (d/f'paired_{p}_s{args.paired_stages}.ttgir').write_text(a.asm['ttgir'])
  ts=times([timed(base),timed(candidate)])
  delta=(old.float()-new.float())
  row=dict(ms=list(map(median,ts)),speedup=median(x/y for x,y in zip(*ts)),bitwise=torch.equal(old.view(torch.uint8),new.view(torch.uint8)),nmse=float(delta.square().sum()/old.float().square().sum()),max_abs=float(delta.abs().max()),resources=[(t.n_regs,t.n_spills,t.metadata.shared) for t in (b,a)],samples=ts)
  (d/f'paired_{p}_s{args.paired_stages}.json').write_text(json.dumps(row,indent=2));print(row)
