import json
import torch
from .sparse_expansion_runtime import refine
from .high_sparse_learn import swap_features,correct
from .high_sparse_gain import score
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.manual_seed(739);rows=[]
    for p in [216,288,410]:
        mass=torch.randn(12,72,p,device='cuda').softmax(-1);feat=torch.randn(12,72,p,8,device='cuda');code=torch.randn(12,p,8,device='cuda',dtype=torch.bfloat16);w=torch.randn(12,14,device='cuda')*.1;own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1);keep=int(p*.1)
        z=mass.clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices;initial=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
        expected,_=correct(initial,score(mass,swap_features(feat,mass,code.float(),initial),w),own)
        for steps in [0,1,2,4]:
            actual,changes=refine(mass,feat,code,w,own,keep,steps)
            assert (actual.sum(-1)==keep).all() and actual.gather(-1,own[...,None]).all()
            assert ((actual^initial).sum(-1)<=2*steps).all()
            if steps==1:assert torch.equal(actual,expected)
            if steps==0:assert torch.equal(actual,initial)
            rows.append(dict(p=p,steps=steps,mean_swaps=float(changes.mean())))
    atomic('artifacts/sparse_expansion_20261031/checks.json',dict(passed=True,rows=rows));print(json.dumps(rows))
if __name__=='__main__':main()
