"""Profile the same captured window in native SDPA and deployed fixed-15 attention."""
import ctypes
import sys
from pathlib import Path
import torch
from control import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,k,v,p,hp,keys,mean,detail,summary,implementation

tiles=(p+15)//16
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
base,_,flags=implementation(library())
candidate,_,new_flags=implementation(ctypes.CDLL('/tmp/jzk_first_formal.so'))
assert not flags.any() and not new_flags.any()
def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
for _ in range(30):
    dense()
    base()
    candidate()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
dense()
base()
candidate()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
