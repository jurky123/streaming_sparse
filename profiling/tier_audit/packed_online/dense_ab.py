import torch,triton
from torch.nn.attention import sdpa_kernel,SDPBackend
import torch.nn.functional as F
torch.manual_seed(701)
for p in (72,144,216,288,360,432):
    q,k,v=[torch.randn(1,n,12,128,device='cuda',dtype=torch.bfloat16).transpose(1,2) for n in [4608,64*p,64*p]]
    for backend in (SDPBackend.FLASH_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(backend):
                def run():return F.scaled_dot_product_attention(q,k,v)
                out=run()
                if backend==SDPBackend.FLASH_ATTENTION: ref=out
                print(p,str(backend),triton.testing.do_bench_cudagraph(run,rep=80),(out-ref).abs().max().item(),flush=True)
        except RuntimeError as e: print(str(e)[:160],flush=True)
