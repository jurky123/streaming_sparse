"""D1536 LayerNorm and modulation with the native intermediate BF16 roundings."""
import json,statistics
import torch
from torch.nn import functional as F
from .profile_route import capture
from .block_pointwise import fused
from .layernorm_mod import apply
from .weights import ROOT

def main():
    torch.manual_seed(701)
    rows=[]
    for factor in (0.01,1.,16.):
        x=torch.randn(1,4608,1536,device='cuda',dtype=torch.bfloat16)*factor
        shift,scale=torch.randn(1,3,2,1536,device='cuda',dtype=x.dtype).chunk(2,dim=2)
        def old():return fused(F.layer_norm(x,(1536,),eps=1e-6),None,scale,shift,True)
        def new():return apply(x,scale,shift,1e-6)
        variants=[capture(f) for f in (old,new)]
        samples=[[],[]]
        for r in range(24):
            for i in (r%2,1-r%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
                for _ in range(50):variants[i][0].replay()
                b.record();b.synchronize()
                if r>=4:samples[i].append(a.elapsed_time(b)/50)
        ms=list(map(statistics.median,samples))
        a,b=[v[1] for v in variants]
        row=dict(scale=factor,ms=ms,speedup=ms[0]/ms[1],mismatch=(a!=b).sum().item(),max_abs=(a.float()-b.float()).abs().max().item())
        print(row,flush=True);rows.append(dict(**row,samples_ms=samples))
    path=ROOT/'artifacts/video_runtime_v1/layernorm_mod_v1';path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
