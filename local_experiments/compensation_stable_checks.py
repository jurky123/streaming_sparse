"""Bitwise reference tests across growing cache sizes and representative scales."""
import json
from pathlib import Path
import torch
from .compensation_kernels import compensate as old
from .compensation_stable_kernels import compensate
from .learned_cache.io import atomic,digest
@torch.no_grad()
def main():
    torch.set_num_threads(4);rows=[]
    for seed in [117,8461,92631]:
        torch.manual_seed(seed)
        for p in [144,216,288,338]:
            n=128;h=12;d=128;cap=338
            q=torch.randn(1,n,h,d,device='cuda',dtype=torch.bfloat16)
            pk=torch.randn(h,cap,4,d,device='cuda')*.4
            code=torch.randn(h,cap,8,device='cuda',dtype=torch.bfloat16)
            basis=torch.randn(h,d,8,device='cuda',dtype=torch.bfloat16)*.1
            center=torch.randn(h,d,device='cuda')*.1;gamma=torch.linspace(0,1,h,device='cuda')
            ids=torch.rand(h,n//64,p,device='cuda').topk(72,-1).indices.sort(-1).values
            lut=torch.cat((ids,torch.arange(p,p+72,device='cuda').expand(h,n//64,72)),-1)[None].int().contiguous()
            exact=torch.randn_like(q);lse=torch.randn(1,h,n,device='cuda')+12
            args=[q,pk,code,lut,exact,lse,basis,center,gamma]
            reference=old(*args,p);predicted=compensate(*args,p,(64,16,4,1,False))
            row=dict(seed=seed,pages=p,output_exact=torch.equal(reference[0],predicted[0]),tail_exact=torch.equal(reference[1],predicted[1]),lse_exact=torch.equal(reference[2],predicted[2]))
            rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/compensation_kernel_stable_20261030/shape_checks.json',dict(rows=rows,passed=all(r['output_exact'] and r['tail_exact'] and r['lse_exact'] for r in rows),code=digest('local_experiments/compensation_stable_kernels.py')))
if __name__=='__main__':main()
