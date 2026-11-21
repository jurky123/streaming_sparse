import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction/vbench_current'))
from quota_mask import quota_mask
import torch,triton
@torch.compile
def old(r,own,c,total):
    p=r.size(-1)
    cr=r[...,-72:].scatter(-1,(own-p+72)[...,None],torch.inf).argsort(dim=-1,descending=True,stable=True).argsort(-1)
    hr=r[...,24:-72].argsort(dim=-1,descending=True,stable=True).argsort(-1)
    return torch.cat((torch.ones_like(r[...,:24],dtype=torch.bool),hr<(total-c[:,None,None]),cr<c[:,None,None]),-1)
for p in (216,288,360,432):
    r=torch.randint(-4,5,(12,72,p),device='cuda').float()
    own=torch.arange(p-72,p,device='cuda').expand(12,-1)
    c=torch.tensor([1,4,9,12,15,18,24,36,42,52,65,72],device='cuda')
    args=r,own,c,72
    assert torch.equal(old(*args),quota_mask(*args))
    print(p,*[triton.testing.do_bench_cudagraph(lambda:f(*args),rep=40) for f in (old,quota_mask)],flush=True)
