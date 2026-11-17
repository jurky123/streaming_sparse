"""Isolated tail fusion contract, measured on a single city_walk event."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .attention import compensate
from .benchmark import measure
from ..learned_cache.io import atomic

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rt,e,identity=inputs();out=rt.compute(e,True)
    fast=torch.compile(compensate,fullgraph=True,options={'triton.cudagraphs':False})
    args=(out[1],e['s'],e['basis'],out[2],e['haar_float'])
    old,x=capture(lambda:compensate(*args));new,y=capture(lambda:fast(*args))
    error=float((x[0]-y[0]).abs().max());nmse=float((x[0]-y[0]).square().sum()/x[0].square().sum());lse=float((x[1]-y[1]).abs().max())
    assert nmse<1e-7 and error<1e-3 and lse<1e-5,(nmse,error,lse)
    report=dict(before=measure(old.replay,25),after=measure(new.replay,25),max_abs=error,nmse=nmse,lse_max_abs=lse,identity=identity)
    atomic('artifacts/video_runtime_v1/hotspot_round3/tail_fusion.json',report);print(json.dumps(report))
if __name__=='__main__':run()
