"""Bounded layout-only softmax prototype: same FP32 operation, different reduction tree."""
import json,math,statistics
import torch
from .weights import ROOT
from ..review_v1.model import Layer
from .grouping_dataflow import balanced
from ..cache_grouping import reorder
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 root=ROOT/'artifacts/video_runtime_v1/grouping_dataflow_v1';s=torch.load(root/'input.pt',weights_only=True)
 m=Layer({n:x.cuda() for n,x in s['model'].items()}).eval().requires_grad_(False)
 k=s['k'].cuda()[0].transpose(0,1).float()[:,None];v=s['v'].cuda()[0].transpose(0,1).float()[:,None]
 kc=k-k.mean(-2,keepdim=True);score=torch.einsum('hptd,hrd->hptr',kc,m.bank)/math.sqrt(128)
 def old():return score.softmax(-2)
 def new():return score.transpose(-2,-1).contiguous().softmax(-1).transpose(-2,-1)
 def capture(fn):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):fn()
  torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  return g,out
 g0,a=capture(old);g1,b=capture(new)
 def time(g):
  times=[]
  for _ in range(20):
   x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True);x.record()
   for _ in range(10):g.replay()
   y.record();y.synchronize();times.append(x.elapsed_time(y)/10)
  return statistics.median(times)
 t0=time(g0);t1=time(g1)
 rms=score.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6);key=score/rms;vp=torch.einsum('hptd,hdr->hptr',v,m.group_v)
 def response(a):
  mean=(a*vp).sum(-2,keepdim=True);r=a*k.shape[-2]*(vp-mean);r=r/r.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
  return (key+.1*m.group_gate.tanh()[:,None,None]*r).squeeze(1)
 fa=response(a);fb=response(b)
 def order(f):
  o=balanced(f,[3,3,2,2,2]);inner=balanced(reorder(f,o).reshape(-1,64,32),[2,2]).reshape(12,72,64)
  return o.reshape(12,72,64).gather(2,inner)
 oa=order(fa);ob=order(fb)
 result=dict(old_ms=t0,new_ms=t1,speedup=t0/t1,softmax_max_abs=float((a-b).abs().max()),feature_max_abs=float((fa-fb).abs().max()),order_equal=torch.equal(oa,ob),different_token_positions=int((oa!=ob).sum()),shape=list(score.shape),stride=list(score.stride()),scope='captured actual layer15 chunk8 clean input; softmax includes transpose materialization; FP32 preserved, reduction tree changes')
 (root/'softmax_prototype.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
