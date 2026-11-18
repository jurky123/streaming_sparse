"""Compare building current summaries in the resident slab with copying them in."""
import json
import statistics
import torch
from .weights import ROOT,load
from .profile_route import capture
from .summary_moments import build
from ..review_v1.ops import Summary


@torch.no_grad()
def main():
    root=ROOT/'artifacts/video_runtime_v1'
    model=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt')[0][15]
    data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
    k,v=(data[n].cuda()[0,:4608].transpose(0,1).reshape(12,72,64,128) for n in ('k','v'))
    fields=('keys','mean','detail','stats')
    rows=[]
    for start in (144,216,288,360):
        variants=[]
        for direct in (False,True):
            slab=Summary(*(torch.full((12,432,*data['summary'][name].shape[2:]),-37.,device='cuda',dtype=torch.bfloat16) for name in fields))
            current=slab.slice(start,start+72)
            def run():
                built=build(model,k,v,True,cuda_pool=True,out=current if direct else None)
                if not direct:
                    for name in fields:getattr(current,name).copy_(getattr(built,name))
                return current
            g,_=capture(run);variants.append((g,current,slab))
        times=[[],[]]
        for r in range(26):
            for i in (r%2,1-r%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
                for _ in range(12):variants[i][0].replay()
                b.record();b.synchronize()
                if r>=2:times[i].append(a.elapsed_time(b)/12)
        ms=list(map(statistics.median,times))
        fields_equal=[torch.equal(getattr(variants[0][1],name),getattr(variants[1][1],name)) for name in fields]
        borders=all(bool((getattr(variants[1][2],name)[:,:start]==-37).all() & (getattr(variants[1][2],name)[:,start+72:]==-37).all()) for name in fields)
        row=dict(pages=start+72,ms=ms,speedup=ms[0]/ms[1],fields_equal=fields_equal,borders=borders)
        print(row,flush=True);rows.append(dict(**row,samples_ms=times))
    folder=root/'summary_window_v1';folder.mkdir(exist_ok=True)
    (folder/'module.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
