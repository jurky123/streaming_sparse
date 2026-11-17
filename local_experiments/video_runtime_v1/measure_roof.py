"""Empirical device throughput references; not interchangeable kernel ceilings."""
import json
from pathlib import Path
import torch
from .profile_route import capture
from .benchmark import measure

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    p=Path('artifacts/video_runtime_v1/hotspot_round8');p.mkdir(parents=True,exist_ok=True)
    x=torch.ones(64*1024*1024,device='cuda');y=torch.empty_like(x)
    g,_=capture(lambda:y.copy_(x));t=measure(g.replay,25);bw=2*x.numel()*x.element_size()/(t['gpu_ms_median']/1000)
    a=torch.randn(2048,2048,device='cuda');b=torch.randn_like(a);o=torch.empty_like(a)
    gg,_=capture(lambda:torch.mm(a,b,out=o));tt=measure(gg.replay,15);flops=2*2048**3/(tt['gpu_ms_median']/1000)
    prop=torch.cuda.get_device_properties(0)
    result=dict(device=prop.name,sm_count=prop.multi_processor_count,copy=t,copy_GBps=bw/1e9,fp32_mm=tt,fp32_mm_TFLOPs=flops/1e12,notes='Copy counts read+write, 256MiB source exceeds L2. FP32 GEMM disables TF32. Empirical references, not guarantees for scalar/nonlinear kernels.')
    (p/'device_roof.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':run()
