import torch
from .kernels import rms_ordered
from .setup import ROOT
from ..common import atomic_json
from ..opt5.probe import capture,timing
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.manual_seed(23);rows=[]
 for n in [512,4680]:
  for scale in [.01,1.,100.]:
   x=(torch.randn(1,n,1536,device='cuda',dtype=torch.bfloat16)*scale);w=torch.randn(1536,device='cuda',dtype=torch.bfloat16);eps=1e-6
   fn=lambda:(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+eps)).to(x.dtype)*w
   y=fn();z=rms_ordered(x,w,eps);g,_=capture(fn);q,_=capture(lambda:rms_ordered(x,w,eps))
   r=dict(n=n,scale=scale,same=torch.equal(y,z),different=int((y!=z).sum()),old_ms=timing(g,100),new_ms=timing(q,100));print(r,flush=True);rows.append(r)
 atomic_json(ROOT/'ordered_probe.json',rows)
if __name__=='__main__':main()
