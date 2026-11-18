"""Same-input pointwise contracts and interleaved CUDA Graph microbenchmarks."""
import json,statistics
from pathlib import Path
import torch
from .block_pointwise import reference,fused
ROOT=Path(__file__).resolve().parents[2]/'artifacts/video_runtime_v1/block_pointwise_v1'
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.manual_seed(701)
 inputs=torch.load(ROOT/'input.pt',weights_only=True);rows=[];contracts=[]
 for name,raw in inputs.items():
  args=[v.cuda() if isinstance(v,torch.Tensor) else v for v in raw]
  for label,scale in [('real',None),('random',1.),('small',.001),('large',100.)]:
   values=[torch.randn_like(v)*scale if scale is not None and isinstance(v,torch.Tensor) else v for v in args]
   # Preserve the actual noncontiguous per-frame modulation view too.
   for stride in (1,6):
    av=list(values)
    for j in (2,3):
     if av[j] is not None:
      v=av[j];back=torch.empty(1,v.shape[1],stride,v.shape[-1],device=v.device,dtype=v.dtype);back[:,:,0:1].copy_(v);av[j]=back[:,:,0:1]
    expected=reference(*av);actual=fused(*av);contracts.append(dict(name=name,input=label,frame_stride=stride,bitwise=torch.equal(expected,actual),max_abs=(expected-actual).abs().max().item()))
  graphs=[]
  for fn in (reference,fused):
   for _ in range(4):fn(*args)
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=fn(*args)
   graphs.append((graph,out))
  for graph,_ in graphs:
   for _ in range(100):graph.replay()
  torch.cuda.synchronize();times=[[],[]]
  for r in range(30):
   for j in ([0,1] if r%2==0 else [1,0]):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    for _ in range(20):graphs[j][0].replay()
    b.record();b.synchronize();times[j].append(a.elapsed_time(b)/20)
  old,new=map(statistics.median,times);rows.append(dict(name=name,before_ms=old,after_ms=new,speedup=old/new,samples_ms=times,elements=args[0].numel()))
 result=dict(contracts=contracts,rows=rows,passed=all(c['bitwise'] for c in contracts))
 (ROOT/'module.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=result['passed'],rows=[{k:v for k,v in r.items() if k!='samples_ms'} for r in rows]),indent=2))
if __name__=='__main__':main()
