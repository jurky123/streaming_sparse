"""Time QKV projection into the active cache window and check its write boundaries."""
import json
import statistics

import torch

from .profile_route import capture
from .rms_mean import project_cached, project_rotary
from .weights import ROOT
from .accepted_tiers import config
from .deployment import configure
from ..broad_common import Engine


@torch.no_grad()
def main():
    configure(config('base',local_assets='/tmp/streaming_sparse-0/runtime'))
    torch.set_num_threads(4)
    torch.manual_seed(701)
    engine=Engine()
    model=engine.wrapper.model
    attn=model.blocks[15].self_attn
    x=torch.randn(1,4608,1536,device='cuda',dtype=attn.q.weight.dtype)
    grid=torch.tensor([[3,32,48]])
    freq=model.freqs.to(x.device).contiguous()
    rows=[]
    for pages in (0,72,144,360):
        start=pages*64
        variants=[]
        for direct in (False,True):
            cache=[x.new_full((1,start+4608+64,12,128),-37.) for _ in range(2)]
            k,v=[buf[:,start:start+4608] for buf in cache]
            def run():
                if direct:
                    return project_cached(attn,x,grid,freq,24,k,v)
                q,keys=project_rotary(attn,x,grid,freq,24)
                values=attn.v(x).view_as(v)
                k.copy_(keys)
                v.copy_(values)
                return q,k,v
            graph,out=capture(run)
            variants.append((graph,out,cache))
        samples=[[],[]]
        for rep in range(22):
            for i in (rep%2,1-rep%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(10):variants[i][0].replay()
                b.record();b.synchronize()
                if rep>=2:samples[i].append(a.elapsed_time(b)/10)
        equal=[torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(variants[0][1],variants[1][1])]
        borders=all(bool((buf[:,:start]==-37).all() & (buf[:,start+4608:]==-37).all()) for buf in variants[1][2])
        ms=list(map(statistics.median,samples))
        row=dict(history_pages=pages,ms=ms,speedup=ms[0]/ms[1],equal_qkv=equal,borders=borders)
        print(row,flush=True);rows.append(dict(**row,samples_ms=samples))
    dest=ROOT/'artifacts/video_runtime_v1/cached_projection_v1'
    dest.mkdir(exist_ok=True)
    (dest/'module.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':
    main()
