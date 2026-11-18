"""Collect instruction delivery and scheduling stalls on Dense, deployed and specialized kernels."""
from pathlib import Path
import sys
import ctypes as C
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,k,v,p,hp,keys,mean,detail,summary,implementation

tiles=(p+15)//16
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
base,ref,flags=implementation(library())
candidate,out,bad=implementation(C.CDLL(f'/tmp/jzk_route_const{p}.so'))
assert not flags.any() and not bad.any() and torch.equal(ref.view(torch.uint8),out.view(torch.uint8))
def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
for _ in range(30):
    dense();base();candidate()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
dense();base();candidate()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
