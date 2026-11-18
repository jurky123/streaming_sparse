"""Measure words first-page copy against the deployed fused kernel."""
import ctypes as C
import json
import os
import sys
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,pk,pv,ids,basis,summary,p,hp,count,keys,mean,detail,timed,times

tiles = (p+15)//16
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
state = torch.zeros(864,device=q.device,dtype=torch.int32)
def make(name):
    lib = library() if name=='base' else C.CDLL(f'/tmp/jzk_route_{name}.so')
    lib.run.argtypes = [C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]
    out = torch.empty_like(q)
    args = tuple(x.data_ptr() for x in (q,pk,pv,ids,out,summary,basis,state))
    def launch(output=out):
        assert lib.run(*args,p,count,torch.cuda.current_stream().cuda_stream)==0
    launch()
    assert not state[:864].any()
    return launch,out
base, ref = make('base')
runs = dict(base=base)
for name in ('words',):
    run, output = make(name)
    assert torch.equal(output.view(torch.uint8),ref.view(torch.uint8))
    runs[name]=run
if os.getenv('PROFILE'):
    for _ in range(30):
        for run in runs.values():run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for run in runs.values():run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
else:
    samples = dict(zip(runs,times([timed(run) for run in runs.values()])))
    results = {}
    for name in runs.keys()-{'base'}:
        pairs=list(zip(samples['base'], samples[name]))
        results[name]=dict(speedup=median(a/b for a,b in pairs),saved_us=1000*median(a-b for a,b in pairs))
    print(p,results)
    path = root/f"words_{p}{os.getenv('RUN_SUFFIX','')}.json"
    path.write_text(json.dumps(dict(results=results,samples_ms=samples,bitwise=True),indent=2))
