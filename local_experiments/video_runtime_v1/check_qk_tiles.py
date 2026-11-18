"""Bounded M16 QK output tiling, preserving four K32 partial-dot additions."""
import json,statistics
from pathlib import Path
import torch,triton
from .tensor_bmm import kernel
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/video_runtime_v1/qk_tiles_v1'
@torch.no_grad()
def main():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
 data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
 q=data['q'].cuda()[0].transpose(0,1).reshape(12,72,64,128).index_select(2,torch.linspace(0,63,16,device='cuda').round().long()).float().flatten(1,2)
 keys=data['summary']['keys'].cuda();rows=[]
 for pages in (216,288,360,432):
  b=keys[:,:pages].float().flatten(1,2).transpose(1,2);h,m,k=q.shape;n=b.shape[-1];graphs=[];outs=[];resources=[]
  for bm,bn in ((32,64),(64,64),(32,128)):
   out=torch.empty(h,m,n,device=q.device,dtype=torch.float32)
   def call():return kernel[(triton.cdiv(m,bm),triton.cdiv(n,bn),h)](q,b,out,m,n,k,*q.stride(),*b.stride(),BM=bm,BN=bn,BK=32,BF16_INPUTS=True,num_warps=4)
   compiled=call();resources.append(dict(bm=bm,bn=bn,registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared))
   for _ in range(4):call()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):call()
   graph.replay();graphs.append(graph);outs.append(out)
  torch.cuda.synchronize();contracts=[dict(bitwise=torch.equal(outs[0],out),max_abs=(outs[0]-out).abs().max().item()) for out in outs]
  for graph in graphs:
   for _ in range(100):graph.replay()
  torch.cuda.synchronize();times=[[],[],[]]
  for r in range(30):
   for i in ([0,1,2] if r%2==0 else [2,1,0]):
    a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
    for _ in range(20):graphs[i].replay()
    z.record();z.synchronize();times[i].append(a.elapsed_time(z)/20)
  med=list(map(statistics.median,times));row=dict(pages=pages,variants=[dict(**rs,**cs,ms=t,speedup=med[0]/t,samples_ms=tm) for rs,cs,t,tm in zip(resources,contracts,med,times)])
  rows.append(row);print(json.dumps(dict(pages=pages,variants=[{k:v for k,v in d.items() if k!='samples_ms'} for d in row['variants']])),flush=True)
 (OUT/'module.json').write_text(json.dumps(dict(rows=rows,source_input='tail_liveness_v1/input.pt; original captured base layer15/chunk8/step0, subsequent accepted changes bitwise'),indent=2))
if __name__=='__main__':main()
