"""Exercise the output staging area on one complete four-warp CTA under racecheck."""
from pathlib import Path
import ctypes as C
import sys
import torch
from local_experiments.video_runtime_v1.fixed15_layout import encode
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import implementation,p,hp,keys,mean,detail,summary
nt=(p+15)//16
encode[(12,nt,42)](keys,mean,detail,summary,p,hp,nt,DIRECT_MASS=True,num_warps=4)
_,_,flags=implementation(C.CDLL('/tmp/jzk_vector_one_cta.so'))
assert flags[0,0].item()==0
torch.cuda.synchronize()
