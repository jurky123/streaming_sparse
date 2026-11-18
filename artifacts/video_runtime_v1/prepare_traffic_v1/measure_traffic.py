"""Profile one deployed prepare launch over the captured native inputs."""
import sys
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import k,v,pk,pv,keys,mean,detail,summary,p,hp
args=[x.data_ptr() for x in (k,v,pk,pv,keys,mean,detail,summary)]
fn=library().prepare_fixed15
stream=torch.cuda.current_stream().cuda_stream
for _ in range(30):
    assert fn(*args,p,hp,stream)==0
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
assert fn(*args,p,hp,stream)==0
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
