import torch, json
from pathlib import Path
times={}
for w in (72,144,216,288,360,432):
    q,k,v=[torch.randn(1,n,12,128,device='cuda',dtype=torch.bfloat16).transpose(1,2) for n in (4608,w*64,w*64)]
    def attention():
        return torch.nn.functional.scaled_dot_product_attention(q,k,v)
    for _ in range(3): attention()
    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(20): attention()
    end.record()
    end.synchronize()
    times[w]=start.elapsed_time(end)/20
Path(__file__).with_name('window.json').write_text(json.dumps(times))
