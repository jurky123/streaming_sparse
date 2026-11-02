import json
from pathlib import Path
import torch
from .compensation_kernels import compensate,reference
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.manual_seed(8461);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
    for p in [144,216,288,338]:
        n=128;h=12;d=128;cap=338
        q=torch.randn(1,n,h,d,device='cuda',dtype=torch.bfloat16)
        pk=torch.randn(h,cap,4,d,device='cuda')*.4
        code=torch.randn(h,cap,8,device='cuda',dtype=torch.bfloat16)
        basis=torch.randn(h,d,8,device='cuda',dtype=torch.bfloat16)*.1
        center=torch.randn(h,d,device='cuda')*.1;gamma=torch.linspace(0,1,h,device='cuda')
        ids=torch.rand(h,n//64,p,device='cuda').topk(72,-1).indices.sort(-1).values
        # Fixed 144-LUT row stride as actual runtime (current IDs not read here).
        lut=torch.cat((ids,torch.arange(p,p+72,device='cuda').expand(h,n//64,72)),-1)[None].int().contiguous()
        exact=torch.randn_like(q);lse=torch.randn(1,h,n,device='cuda')+12
        actual=compensate(q,pk,code,lut,exact,lse,basis,center,gamma,p);expected=reference(q,pk,code,lut,exact,lse,basis,center,gamma,p)
        torch.testing.assert_close(actual[1],expected[1],atol=3e-5,rtol=2e-4)
        torch.testing.assert_close(actual[2],expected[2],atol=3e-5,rtol=2e-5)
        # BF16 output may differ by one rounding boundary; compare to FP32-scale tolerance.
        torch.testing.assert_close(actual[0].float(),expected[0].float(),atol=.016,rtol=.008)
        assert torch.equal(actual[0][:,:,0],exact[:,:,0]),'gamma=0 must exactly preserve output'
        rows.append(dict(pages=p,max_tail_error=(actual[1]-expected[1]).abs().max().item(),max_log2_partition_error=(actual[2]-expected[2]).abs().max().item(),output_nmse=((actual[0].float()-expected[0].float()).square().mean()/expected[0].float().square().mean()).item()))
    root=Path('artifacts/compensation_probe_20261030');atomic(root/'kernel_checks.json',dict(passed=True,rows=rows,scope='synthetic kernel formula equivalence only; real-QKV and closed-loop audits still required'))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
