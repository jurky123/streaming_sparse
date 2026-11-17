"""Measured throughput references for the current module cost table."""
import json
from pathlib import Path
import torch
from .profile_route import capture
from .benchmark import measure
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows={}
    for name,dtype,n in [('fp32',torch.float32,2048),('bf16',torch.bfloat16,4096),('fp64',torch.float64,2048)]:
        a=torch.randn(n,n,device='cuda',dtype=dtype);b=torch.randn_like(a);o=torch.empty_like(a)
        g,_=capture(lambda:torch.mm(a,b,out=o));t=measure(g.replay,25)
        rows[name]=dict(ms=t['gpu_ms_median'],tflops=2*n**3/t['gpu_ms_median']/1e9,shape=n)
    x=torch.ones(64*1024*1024,device='cuda');y=torch.empty_like(x)
    g,_=capture(lambda:y.copy_(x));t=measure(g.replay,25)
    rows['copy']=dict(ms=t['gpu_ms_median'],gbps=2*x.numel()*4/t['gpu_ms_median']/1e6)
    rows['device']=torch.cuda.get_device_name();rows['notes']='Empirical large GEMM/copy reference, not hardware peak or an achievable guarantee for small nonlinear kernels.'
    Path('artifacts/video_runtime_v1/lazy_video_v1/device_roof.json').write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':main()
