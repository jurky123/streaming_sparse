"""Check exact threshold and index tie breaks independently of real-data probes."""
import torch,triton as tr
from .radix import top
from .setup import ROOT
from ..opt17.route import _top
from ..common import atomic_json
@torch.no_grad()
def main():
 torch.manual_seed(314);cases=[]
 for b in [74,296,676,1183,2051]:
  for kind in ['normal','ties','zeros','wide']:
   s=torch.randn((2,7,b),device='cuda')
   if kind=='ties':s=s.round()
   elif kind=='zeros':s.zero_()
   elif kind=='wide':s=s*torch.logspace(-20,20,b,device='cuda')[None,None]
   for c in sorted({1,b,b//5,(b+1)//2}):
    o=top(s,c,8,4);ref=torch.empty_like(o);_top[(14,)](s,ref,14,b,c,tr.next_power_of_2(b),num_warps=4)
    assert torch.equal(o,ref),(b,kind,c)
    # Preserve the parent's IEEE bit ordering, including -0.0 before +0.0.
    bits=s.view(torch.int32).long()&0xffffffff
    ordered=torch.where((bits&0x80000000)!=0,(~bits)&0xffffffff,bits^0x80000000)
    expected=torch.argsort(ordered,dim=-1,descending=True,stable=True)[...,:c].sort(-1).values
    assert torch.equal(o,expected),(b,kind,c,'torch reference')
    cases.append(dict(b=b,kind=kind,keep=c))
 atomic_json(ROOT/'radix_correctness.json',dict(passed=True,cases=cases));print('passed',len(cases),flush=True)
if __name__=='__main__':main()
