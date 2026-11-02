import torch
from pathlib import Path
from .improve_learned_kernels import route
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(72);torch.backends.cuda.matmul.allow_tf32=False
    q=torch.randn(1,4608,12,128,device='cuda',dtype=torch.bfloat16)
    alpha=torch.rand(12,device='cuda');alpha[::3]=0
    rows=[]
    for p in [144,216,288,338]:
        pk=torch.randn(12,p,4,128,device='cuda');center=torch.randn(12,128,device='cuda')*.2
        logs=torch.rand(12,p,4,device='cuda')-.5
        x=q[0].transpose(0,1).float().reshape(12,72,64,128);mean=x.mean(2);var=x.var(2,unbiased=False)
        for use_center,use_count in [(False,False),(True,False),(True,True)]:
            c=center if use_center else None;lc=logs if use_count else None
            actual=route(q,pk,alpha,c,lc)
            z=(mean@pk.flatten(1,2).transpose(1,2)/128**.5).reshape(12,72,p,4)
            shifted=pk if c is None else pk-c[:,None,None]
            z+=alpha[:,None,None,None]*(var@shifted.square().flatten(1,2).transpose(1,2)/256).reshape_as(z)
            if lc is not None:z+=lc[:,None]
            expected=z.logsumexp(-1);torch.testing.assert_close(actual,expected,atol=8e-4,rtol=5e-4)
            rows.append(dict(pages=p,center=use_center,counts=use_count,max_abs=(actual-expected).abs().max().item()))
    atomic(Path('artifacts/rollout_improve_20261030/score_checks.json'),dict(passed=True,rows=rows))
if __name__=='__main__':main()
