import json
from pathlib import Path
import torch
from statistics import median
def measure(fn):
    fn()
    pairs=[]
    for _ in range(30):
        a,b=[torch.cuda.Event(enable_timing=True) for i in range(2)]
        a.record()
        fn()
        b.record()
        pairs.append((a,b))
    torch.cuda.synchronize()
    return median(a.elapsed_time(b) for a,b in pairs)
a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)
b=torch.randn_like(a)
c=torch.empty_like(a)
gemm=2*4096**3/measure(lambda:torch.mm(a,b,out=c))/1e9
a=torch.ones(64*1024*1024,device='cuda')
b=torch.empty_like(a)
bw=8*a.numel()/measure(lambda:b.copy_(a))/1e6
Path(__file__).with_name('roof.json').write_text(json.dumps(dict(tflops=gemm,GBps=bw)))
