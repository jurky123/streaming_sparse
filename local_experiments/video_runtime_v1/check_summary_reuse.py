"""Reuse existing repository summary kernels on the full-coverage C weights."""
import json
from pathlib import Path
import torch
from .weights import load,ROOT
from .profile_route import capture
from .benchmark import measure
from ..review_v1.ops import SummaryOps,haar
from ..review_v1.fused_summary import build as fused
from ..review_v1.split_summary import build as split

@torch.no_grad()
def run(native=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model,_,_,_,identity=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt')
    path=next((ROOT/'artifacts/full_coverage_v1/data/city_walk_s701').glob('c8_l15_s0_h0_t*.pt'));ev=torch.load(path,weights_only=True);kv=torch.load(path.parent/ev['kv_file'],weights_only=True)
    k=kv['k'][-72:].cuda().bfloat16()[None].expand(12,-1,-1,-1).contiguous();v=kv['v'][-72:].cuda().bfloat16()[None].expand_as(k).contiguous();ha=haar(k);layer=model[15]
    if native:
        k=k.flatten(1,2).transpose(0,1).contiguous().transpose(0,1).reshape(12,72,64,128)
        v=v.flatten(1,2).transpose(0,1).contiguous().transpose(0,1).reshape(12,72,64,128)
    rows=[]
    for cheap in (True,False):
        old=lambda:SummaryOps.build(layer,k,v,cheap,haar_basis=ha)
        go,x=capture(old);before=measure(go.replay,25)
        for name,fn in [('fused',fused),('split',split)]:
            gn,y=capture(lambda:fn(layer,k,v,cheap));errors={}
            for field in ('keys','mean','detail','stats'):
                a,b=getattr(x,field).float(),getattr(y,field).float();assert torch.isfinite(b).all()
                errors[field]=dict(nmse=float((a-b).square().sum()/a.square().sum().clamp_min(1e-30)),max_abs=float((a-b).abs().max()))
            after=measure(gn.replay,25);rows.append(dict(cheap=cheap,method=name,before=before,after=after,errors=errors));print(cheap,name,before['gpu_ms_median'],after['gpu_ms_median'],errors,flush=True)
    dest=Path('artifacts/video_runtime_v1/hotspot_round7');dest.mkdir(parents=True,exist_ok=True);(dest/('summary_native.json' if native else 'summary_reuse.json')).write_text(json.dumps(dict(rows=rows,identity=identity),indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');run(p.parse_args().native)
