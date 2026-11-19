import json,statistics,torch
from pathlib import Path
torch.backends.cuda.matmul.allow_tf32=False
d=Path(__file__).parent
def bench(fn):
    for _ in range(4):fn()
    samples=[]
    for _ in range(9):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):fn()
        b.record();b.synchronize();samples.append(a.elapsed_time(b)/5)
    return statistics.median(samples)
r={}
for t in (torch.bfloat16,torch.float32,torch.float64):
    n=2048
    x=torch.randn((n,n),device='cuda',dtype=t);y=torch.randn_like(x);z=torch.empty_like(x)
    tm=bench(lambda:torch.mm(x,y,out=z))
    r[str(t)]=dict(ms=tm,tflops=2*n**3/tm/1e9)
x=torch.empty(128*2**20,device='cuda',dtype=torch.float32);y=torch.empty_like(x)
tm=bench(lambda:y.copy_(x));r['copy']=dict(ms=tm,GBps=2*x.numel()*4/tm/1e6)
(d/'roof.json').write_text(json.dumps(r,indent=2))
