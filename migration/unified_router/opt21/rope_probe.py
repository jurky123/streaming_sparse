import torch,json
from .setup import ROOT
from ..common import atomic_json
from ..opt5.probe import capture,timing
from .rope import rope
from wan.modules.causal_model import causal_rope_apply
from wan.modules.model import rope_params
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.manual_seed(21)
 freqs=torch.cat([rope_params(1024,44),rope_params(1024,42),rope_params(1024,42)],dim=1).cuda();grid=torch.tensor([[3,30,52]])
 rows=[]
 for start in [0,3,6,9]:
  x=torch.randn(1,4680,12,128,device='cuda',dtype=torch.bfloat16)
  expected=causal_rope_apply(x,grid,freqs,start);actual=rope(x,grid,freqs,start)
  assert torch.equal(expected,actual),(start,float((actual.float()-expected.float()).abs().max()),int((actual!=expected).sum()))
  g,_=capture(lambda:causal_rope_apply(x,grid,freqs,start));q,_=capture(lambda:rope(x,grid,freqs,start))
  row=dict(start=start,bit_identical=True,old_ms=timing(g,100),new_ms=timing(q,100));rows.append(row);print(row,flush=True)
 atomic_json(ROOT/'rope_probe.json',rows)
if __name__=='__main__':main()
