"""Check the custom mean tree and RMS output against the accepted dataflow path."""
import json,statistics
from pathlib import Path
import torch,triton
from .rms_dataflow import fused as reference,_finish
from .rms_mean import fused,mean_square
ROOT=Path(__file__).resolve().parents[2]/'artifacts/video_runtime_v1/rms_mean_v1'
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.manual_seed(701);inputs=torch.load(ROOT.parent/'rms_dataflow_v1/input.pt',weights_only=True);rows=[];contracts=[];kernels={}
 for name,values in inputs.items():
  x=values['x'].cuda();w=values['weight'].cuda();eps=values['eps']
  for label,scale in [('real',None),('random',1.),('small',.001),('large',100.),('zero',0.),('tiny',1.e-15),('huge',1.e15)]:
   xx=x if scale is None else torch.randn_like(x)*scale
   for epsilon in (eps,1.e-5):
    expected=reference(xx,w,epsilon);actual=fused(xx,w,epsilon)
    refsq=xx.float().pow(2)
    contracts.append(dict(name=name,input=label,eps=epsilon,mean_bitwise=torch.equal(mean_square(xx),refsq.mean(-1,keepdim=True)),output_bitwise=torch.equal(expected,actual),max_abs=(expected-actual).abs().max().item()))
  graphs=[]
  for fn in (reference,fused):
   for _ in range(4):fn(x,w,eps)
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=fn(x,w,eps)
   graphs.append((g,out))
  for g,_ in graphs:
   for _ in range(100):g.replay()
  torch.cuda.synchronize();times=[[],[]]
  for r in range(30):
   for j in ([0,1] if r%2==0 else [1,0]):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    for _ in range(20):graphs[j][0].replay()
    b.record();b.synchronize();times[j].append(a.elapsed_time(b)/20)
  old,new=map(statistics.median,times);rows.append(dict(name=name,shape=list(x.shape),before_ms=old,after_ms=new,speedup=old/new,samples_ms=times))
  inv=torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps);out=torch.empty_like(x)
  tail=_finish[(triton.cdiv(x.numel(),1024),)](x,inv,w,out,x.numel(),x.shape[-1],1024,num_warps=4,enable_fp_fusion=False)
  for label,compiled in [('finish',tail)]:
   kernels[label]=dict(registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared,fma='fma.rn' in compiled.asm['ptx']);(ROOT/f'{label}.ptx').write_text(compiled.asm['ptx'])
 passed=all(c['output_bitwise'] and c['mean_bitwise'] for c in contracts)
 (ROOT/'module.json').write_text(json.dumps(dict(rows=rows,contracts=contracts,kernels=kernels,passed=passed),indent=2))
 print(json.dumps(dict(passed=passed,rows=[{k:v for k,v in r.items() if k!='samples_ms'} for r in rows]),indent=2))
if __name__=='__main__':main()
