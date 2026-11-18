"""Hold a captured window fixed and vary only the length of its exact-page list."""
import ctypes
import json
import os
import sys
from pathlib import Path
from statistics import median

import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode

root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,k,v,pk,pv,ids,summary,basis,p,hp,count,keys,mean,detail,timed,times

nt=(p+15)//16
encode[(12,nt,42)](keys,mean,detail,summary,p,hp,nt,DIRECT_MASS=True,num_warps=4)
lib=library()
def make(route):
    out=torch.empty_like(q)
    flags=torch.empty((12,72),device=q.device,dtype=torch.int32)
    tensors=(q,pk,pv,route,out,summary,basis,flags)
    def launch():
        assert lib.run(*(x.data_ptr() for x in tensors),p,route.shape[-1],torch.cuda.current_stream().cuda_stream)==0
    launch()
    assert not flags.any()
    return launch

def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
used=torch.zeros((*ids.shape[:-1],p),device=q.device,dtype=torch.bool)
used.scatter_(-1,ids.long(),True)
free=torch.arange(p,device=q.device).expand_as(used)[~used].reshape(*ids.shape[:-1],p-count)
route=torch.cat((ids,free.to(ids.dtype)),dim=-1)
lengths=[32,40,48,56,64]
runs=[make(route[...,:c].contiguous()) for c in lengths]
x,*samples=times([timed(dense)]+[timed(f) for f in runs])
dense_rate=4*q.shape[1]*12*128*p*64/1e9/median(x)
output=dict(p=p,deployed_count=count,dense_ms=median(x),dense_tflops=dense_rate,samples={})
for c,y in zip(lengths,samples):
    reference=(192*c+102*(nt-1))*3456*4096/1e9/dense_rate
    output['samples'][c]=dict(ms=median(y),reference_ms=reference,raw=y)
(root/f"window_{p}{os.getenv('RUN_SUFFIX','')}.json").write_text(json.dumps(output,indent=2))
