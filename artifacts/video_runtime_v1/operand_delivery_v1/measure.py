"""Alternate paired CUDA Graph launches on the same captured input."""
import json
import os
import sys
import torch
import ctypes as C
from statistics import median
from local_experiments.video_runtime_v1.fixed15_attention import library
from capture import implementation, p, root, timed, times

old,a,af=implementation(C.CDLL('/tmp/jzk_delivery_control.so'))
name=os.environ['CANDIDATE']
new,b,bf=implementation(library() if name=='installed' else C.CDLL('/tmp/jzk_delivery_'+name+'.so'))
assert not af.any() and torch.equal(af,bf)
bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8))
delta=b.float()-a.float()
nmse=float(delta.square().sum()/a.float().square().sum())
assert bitwise or (name in ("tensor_mass","fused_center","anchor_center","floor_center") and torch.isfinite(delta).all())
if os.getenv('COUNTERS'):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    old();new()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    sys.exit()
x,y=times([timed(old),timed(new)])
result=dict(p=p,bitwise=bitwise,nmse=nmse,max_abs=float(delta.abs().max()),changed=int((a!=b).sum()),base_ms=median(x),candidate_ms=median(y),paired_speedup=median(i/j for i,j in zip(x,y)),samples=[x,y])
(root/f'{name}_{p}.json').write_text(json.dumps(result,indent=2))
print(result['paired_speedup'])
