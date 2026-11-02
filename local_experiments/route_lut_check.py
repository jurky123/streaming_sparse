import torch,triton as tr,json
from .route_fused_kernels import _lut
from .learned_cache.io import atomic
from pathlib import Path
@torch.no_grad()
def main():
    torch.manual_seed(31);rows=[]
    for p in [216,288,360,410]:
        k=int(p*.1+1e-6);own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1)
        for mode in ['random','ties','all_equal']:
            mass=torch.randn(12,72,p,device='cuda');mass.scatter_(-1,own[...,None],float('inf'));ids=mass.topk(k,-1).indices
            mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
            s=torch.randn_like(mass)
            if mode=='ties':s=s.round()
            if mode=='all_equal':s.zero_()
            rem=s.masked_fill(~mask,float('inf'));rem.scatter_(-1,own[...,None],float('inf'));low,drop=rem.min(-1);high,add=s.masked_fill(mask,-float('inf')).max(-1)
            ref=torch.where((ids==drop[...,None])&(high>low)[...,None],add[...,None],ids).sort(-1).values[None].int()
            out=torch.empty_like(ref);_lut[(864,)](s,mask,ids,out,p,k,tr.next_power_of_2(p),tr.next_power_of_2(k),num_warps=4)
            assert torch.equal(ref,out);rows.append(dict(p=p,mode=mode,passed=True))
    atomic(Path('artifacts/route_kernel_20261031/lut_edge_checks.json'),rows);print(json.dumps(rows))
if __name__=='__main__':main()
