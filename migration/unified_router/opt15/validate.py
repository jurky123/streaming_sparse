"""Check packed bits, stable ties and nonmultiple query/page tails."""
import torch,triton as tr
from .bitset_fast import _pack,bitset_order
from .setup import ROOT
from ..common import atomic_json
@torch.no_grad()
def main():
 torch.manual_seed(19);rows=[]
 for n,b in [(31,74),(32,148),(33,296),(65,74),(129,296)]:
  for kind in ['random','identical']:
   m=torch.rand((2,n,b),device='cuda')<.15 if kind=='random'else torch.ones((2,n,b),device='cuda',dtype=torch.bool)
   words=tr.cdiv(b,32);bits=torch.empty((2,n,words),device='cuda',dtype=torch.int32);_pack[(tr.cdiv(n,32),2)](m,bits,n,b,words,tr.next_power_of_2(b),num_warps=4)
   p=torch.arange(b,device='cuda');decoded=((bits.long()[:,:,p//32]>>(p%32)[None,None])&1).bool();assert torch.equal(m,decoded)
   for method in ['BitsetFast_64_16_1','BitsetFast_128_32_2']:
    ix=bitset_order(m,method);assert torch.equal(ix.sort(-1).values,torch.arange(n,device='cuda').expand(2,n));assert torch.equal(ix,bitset_order(m,method))
    if kind=='identical':assert torch.equal(ix,torch.arange(n,device='cuda').expand(2,n))
    rows.append(dict(n=n,pages=b,kind=kind,method=method,packed_bits_exact=True,permutation_valid=True,repeat_identical=True))
 atomic_json(ROOT/'validation.json',dict(cases=rows,total=len(rows)));print('passed',len(rows),flush=True)
if __name__=='__main__':main()
