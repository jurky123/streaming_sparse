"""Interleave current sparse and native Dense attention on each real captured shape."""
from pathlib import Path
from statistics import median
import ctypes
import json
import os
import sys

import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode

root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,k,v,keys,mean,detail,summary,p,hp,count,timed,times,implementation

nt=(p+15)//16
encode[(12,nt,42)](keys,mean,detail,summary,p,hp,nt,DIRECT_MASS=True,num_warps=4)
sparse,_,flags=implementation(ctypes.CDLL('/tmp/jzk_anchor_control.so'))
assert not flags.any()
def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
if os.getenv('PROFILE_RELEASE'):
    installed,_,flags=implementation(ctypes.CDLL('/tmp/jzk_anchor_control.so'))
    assert not flags.any()
    for _ in range(25):
        dense(); installed()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    dense(); installed()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    sys.exit()
if os.getenv('VECTOR'):
    vector,_,vf=implementation(ctypes.CDLL('/tmp/jzk_anchor_half.so'))
    assert not vf.any()
    x,y,z=times([timed(dense),timed(sparse),timed(vector)])
else:
    x,y=times([timed(dense),timed(sparse)])
dense_work=4*q.shape[1]*12*128*p*64/1e9
sparse_work=(192*count+102*(nt-1))*3456*4096/1e9
d,s=median(x),median(y)
ref=d*sparse_work/dense_work
row=dict(p=p,exact=count,dense_ms=d,main_ms=s,dense_tflops=dense_work/d,main_tflops=sparse_work/s,
         throughput_fraction=ref/s,reference_ms=ref,samples=[x,y])
if os.getenv('VECTOR'):
    row.update(vector_ms=median(z),vector_saving_us=1000*median(i-j for i,j in zip(y,z)),vector_gap_us=1000*(median(z)-d*(sparse_work-32*3456*4096/1e9)/dense_work),
               candidate_work_gflop=sparse_work-32*3456*4096/1e9,candidate_reference_ms=d*(sparse_work-32*3456*4096/1e9)/dense_work,vector_tflops=(sparse_work-32*3456*4096/1e9)/median(z),vector_samples=z)
(root/f"dense{os.getenv('RUN_SUFFIX','')}_{p}.json").write_text(json.dumps(row,indent=2))
