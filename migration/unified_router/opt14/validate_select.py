"""Check associative reduction against same-warp serial semantics, including ties and missing pools."""
import torch,triton as tr
from ..opt9.select import _packed_select
from ..opt10.compact_route import CompactMomentRoute
from ..common import atomic_json
from .select import select
from .setup import ROOT,PARENT
@torch.no_grad()
def main():
 torch.manual_seed(47);rows=[]
 def check(x,o,beta,name):
  h,n,b=x.shape;m=torch.empty_like(x,dtype=torch.bool);r=torch.empty((h*n,8),device=x.device,dtype=torch.int64);a=torch.empty_like(r)
  _packed_select[(h*n,)](x,m,o,r,a,n,b,tr.next_power_of_2(b),beta,'packed32',num_warps=1,enable_fp_fusion=False)
  actual=select(x,o,beta,1);assert all(torch.equal(u,v)for u,v in zip(actual,(m,r,a))),name
  rows.append(dict(case=name,mask_and_candidates_exact=True))
 for f in sorted((PARENT/'captured').glob('*.pt')):
  d=torch.load(f,map_location='cuda',weights_only=False);route=CompactMomentRoute(batch=4680);state=route.begin(d['prepared'],d['sketch']);score,_=route.score(d['q'],state);check(score,d['sketch'].origins,d['beta'],f.name)
 for b in [148,222,296]:
  o=torch.cat([torch.zeros(b-74,device='cuda'),torch.ones(74,device='cuda')]);rnd=torch.randn(2,129,b,device='cuda')
  for name,x in [('random',rnd),('all_zero',torch.zeros_like(rnd)),('ties',torch.arange(b,device='cuda').float().remainder(5)[None,None].expand_as(rnd).contiguous()),('near_constant',torch.ones_like(rnd)+rnd*1e-7)]:
   for pool,origins in [('mixed',o),('all_history',torch.zeros_like(o)),('all_current',torch.ones_like(o))]:check(x,origins,.7,f'{b}_{name}_{pool}')
 atomic_json(ROOT/'select_validation.json',dict(cases=rows,total=len(rows),comparison='serial num_warps=1 vs top3 num_warps=1; isolates reduction correctness from mean/std rounding changes versus deployed num_warps=4'))
 print('passed',len(rows),flush=True)
if __name__=='__main__':main()
