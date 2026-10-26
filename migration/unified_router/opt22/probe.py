import torch,json
from .setup import ROOT
from .kernels import modulate,residual,rms
from ..opt5.probe import capture,timing
from ..common import atomic_json

@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.manual_seed(22);rows=[]
 x=torch.randn(1,4680,1536,device='cuda',dtype=torch.bfloat16);a=torch.randn_like(x)
 e=torch.randn(1,3,6,1536,device='cuda',dtype=torch.bfloat16).chunk(6,dim=2);weight=torch.randn(1536,device='cuda',dtype=torch.bfloat16);eps=1e-6
 tests=[('modulation',lambda:(x.unflatten(1,(3,1560))*(1+e[1])+e[0]).flatten(1,2),lambda:modulate(x,e[1],e[0])),
 ('residual',lambda:x+(a.unflatten(1,(3,1560))*e[2]).flatten(1,2),lambda:residual(x,a,e[2]))]
 for mode in ['native_stats','fused']:
  tests.append(('rms_'+mode,lambda:(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+eps)).to(x.dtype)*weight,lambda mode=mode:rms(x,weight,eps,mode)))
 for name,ref,new in tests:
  y=ref();z=new();g,_=capture(ref);q,_=capture(new)
  row=dict(name=name,bit_identical=torch.equal(y,z),different=int((y!=z).sum()),nmse=float((y.float()-z.float()).square().sum()/y.float().square().sum()),max_abs=float((y.float()-z.float()).abs().max()),old_ms=timing(g,100),new_ms=timing(q,100));rows.append(row);print(row,flush=True)
 atomic_json(ROOT/'probe.json',rows)
if __name__=='__main__':main()
