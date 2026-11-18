"""Compare native Dense, deployed CUDA and anchor-priority CUDA in rotated order."""
import ctypes
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
from capture import q,k,v,pk,pv,keys,mean,detail,summary,p,hp,count,ids,timed,times,implementation

tiles=(p+15)//16
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
torch.testing.assert_close(ids[...,1:25],torch.arange(24,device=q.device,dtype=ids.dtype).expand_as(ids[...,1:25]))
base,out,flags=implementation(library())
cand,new,new_flags=implementation(ctypes.CDLL('/tmp/jzk_anchor_cache.so'))
assert not flags.any() and torch.equal(flags,new_flags) and torch.equal(out.view(torch.uint8),new.view(torch.uint8))
def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))

if os.getenv('COUNTERS'):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    base()
    cand()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    sys.exit()

# Include the deployed half-width first compensation QK.
flop=(192*count+102*(tiles-1)-32)*3456*4096
dense_flop=4*q.shape[1]*12*128*p*64
d,b,c=times([timed(dense),timed(base),timed(cand)])
row=dict(p=p,bitwise=True,dense_ms=median(d),base_ms=median(b),candidate_ms=median(c),
         work_gflop=flop/1e9,dense_reference_ms=median(d)*flop/dense_flop,
         speedup=median(x/y for x,y in zip(b,c)),samples=[d,b,c])
(root/f"p{p}{os.getenv('RUN_SUFFIX','')}.json").write_text(json.dumps(row,indent=2))
print({k:v for k,v in row.items() if k!='samples'})
