"""Cross selected pages and executed tail tiles on one physical P432 window."""
import ctypes as C
import json
import os
import sys
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,k,v,pk,pv,ids,summary,basis,p,hp,count,keys,mean,detail,timed,times,implementation
assert (p,count)==(432,64)
nt=(p+15)//16
encode[(12,nt,42)](keys,mean,detail,summary,p,hp,nt,DIRECT_MASS=True,num_warps=4)
base,expected,flags=implementation(library())
lib=C.CDLL('/tmp/jzk_phase_factor.so')
lib.run.argtypes=[C.c_void_p]*8+[C.c_int]*3+[C.c_void_p]
def make(c,blocks):
    order=ids[...,:c].contiguous()
    out=torch.empty_like(q)
    bad=torch.empty_like(flags)
    inputs=(q,pk,pv,order,out,summary,basis,bad)
    def launch():
        assert lib.run(*(t.data_ptr() for t in inputs),p,c,blocks+1,torch.cuda.current_stream().cuda_stream)==0
    launch()
    assert not bad.any() and torch.isfinite(out).all()
    if (c,blocks)==(64,26):
        assert torch.equal(expected.view(torch.uint8),out.view(torch.uint8))
    return launch

def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
runs={f'{c}_{b}':make(c,b) for c in (32,48,64) for b in (6,16,26)}
if os.getenv('PROFILE'):
    for _ in range(20):dense();base()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    dense();base()
    for f in runs.values():f()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
else:
    x,y,*rest=times([timed(dense),timed(base)]+[timed(f) for f in runs.values()])
    records=dict(zip(runs,rest))
    records.update(dense=x,base=y)
    (root/'events.json').write_text(json.dumps(records,indent=2))
