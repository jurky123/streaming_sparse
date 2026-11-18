"""Pair K normalization/rotation plus packed input preparation on a captured window."""
import ctypes
import json
import os
import sys
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1 import rms_mean
from local_experiments.video_runtime_v1.fixed15_attention import library

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import k,v,pk,pv,keys,mean,detail,summary,hp,p,times,timed

variant=os.environ.get('VARIANT','direct')
miss=os.environ.get('CACHE','hit')=='miss'
x=k[:,-4608:].clone().contiguous()
weight=torch.linspace(.8,1.2,1536,device=k.device,dtype=k.dtype)
freq=torch.polar(torch.ones((128,64),dtype=torch.double,device=k.device),
    torch.arange(128,device=k.device)[:,None]*torch.exp(-torch.arange(64,dtype=torch.double,device=k.device)[None,:]/8))
old=library()
native=rms_mean.library().rms_rope
dual=rms_mean.library().rms_rope_pack if variant=='release' else ctypes.CDLL('/tmp/jzk_ropepack_'+variant+'.so').rms_rope
dual.argtypes=native.argtypes[:-1]+[ctypes.c_void_p,ctypes.c_void_p]
dual.restype=None
vonly=old if variant=='release' else ctypes.CDLL('/tmp/jzk_ropepack_vonly.so')
def setup(candidate):
    output=k.clone()
    packed_k,packed_v=pk.clone(),pv.clone()
    current=output[:,-4608:]
    packed=packed_k.view(p,-1)[-72:]
    dst=summary.clone()
    args=(x.data_ptr(),weight.data_ptr(),current.data_ptr(),4608,1e-6,freq.data_ptr(),32,48,8)
    pre=[t.data_ptr() for t in (output,v,packed_k,packed_v,keys,mean,detail,dst)]
    def rope():
        stream=torch.cuda.current_stream().cuda_stream
        if candidate: dual(*args,packed.data_ptr(),stream)
        else: native(*args,stream)
    lib=vonly if candidate else old
    kernel=lib.prepare_fixed15 if miss else lib.prepare_current_fixed15
    kernel.argtypes=old.prepare_fixed15.argtypes
    def prepare():
        stream=torch.cuda.current_stream().cuda_stream
        if candidate and variant=='release':
            assert old.prepare_v_fixed15(*pre,p,hp,not miss,stream)==0
        else:
            assert kernel(*pre,p,hp,stream)==0
    def full():
        rope()
        prepare()
    packed.fill_(float('nan'))
    full()
    return (output,packed_k,packed_v,dst),rope,prepare,full
a,r0,p0,f0=setup(False)
b,r1,p1,f1=setup(True)
for old_tensor,new_tensor in zip(a,b):
    assert torch.equal(old_tensor.view(torch.uint8),new_tensor.view(torch.uint8))
rows=times([timed(f0),timed(f1),timed(r0),timed(r1),timed(p0),timed(p1)])
def pair(x,y):
    return dict(base_us=1000*median(x),candidate_us=1000*median(y),
                saving_us=1000*median(a-b for a,b in zip(x,y)),speedup=median(a/b for a,b in zip(x,y)),samples=[x,y])
result={k:pair(*rows[i:i+2]) for k,i in [('combined',0),('rope',2),('prepare',4)]}
result['bitwise']=True
(root/f'{variant}_{p}{"_miss" if miss else ""}.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v['speedup'] for k,v in result.items() if isinstance(v,dict)})
