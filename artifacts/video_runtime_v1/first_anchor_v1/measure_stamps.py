"""Compare identical full fused outputs using measured, cached and reordered anchors."""
import ctypes as C
import json
import os
import sys
from pathlib import Path
from statistics import median
import torch
from control import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,pk,pv,ids,basis,summary,p,hp,count,keys,mean,detail,timed,times

tiles = (p+15)//16
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
state = torch.zeros(864+2*864*(count+2),device=q.device,dtype=torch.int32)
state[864:].view(torch.float32).fill_(float('nan'))
def make(name):
    lib = library() if name=='base' else C.CDLL(f'/tmp/jzk_first_{name}.so')
    lib.run.argtypes = [C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]
    out = torch.empty_like(q)
    args = tuple(x.data_ptr() for x in (q,pk,pv,ids,out,summary,basis,state))
    def launch(output=out):
        assert lib.run(*args,p,count,torch.cuda.current_stream().cuda_stream)==0
    launch()
    assert not state[:864].any()
    return launch,out
base, ref = make('base')
run, out=make('stamps')
assert torch.equal(out.view(torch.uint8),ref.view(torch.uint8))
for _ in range(30):run()
readings=[]
ticks=state[864:].view(torch.int64).view(864,count+2)
for _ in range(15):
    run()
    diff=ticks.diff(dim=1)
    assert diff.min()>0
    readings.append(diff.double().mean(0).tolist())
cycles=list(map(median,zip(*readings)))
x,y=times([timed(base),timed(run)])
result=dict(prologue=cycles[0],first_page=cycles[1],interior_page=median(cycles[2:-1]),last_page=cycles[-1],
            cycles=cycles,bitwise=True,wall_ms=[median(x),median(y)])
(root/f'stamps_{p}.json').write_text(json.dumps(result,indent=2))
