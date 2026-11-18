"""Measure words first-page copy against the deployed fused kernel."""
import ctypes as C
import json
import os
import sys
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode
root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,pk,pv,ids,basis,summary,p,hp,count,keys,mean,detail,timed,times

tiles = (p+15)//16
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
state = torch.zeros(864,device=q.device,dtype=torch.int32)
def make(name):
    lib = library() if name=='base' else C.CDLL(f'/tmp/jzk_route_{name}.so')
    lib.run.argtypes = [C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]
    out = torch.empty_like(q)
    args = tuple(x.data_ptr() for x in (q,pk,pv,ids,out,summary,basis,state))
    def launch(output=out):
        assert lib.run(*args,p,count,torch.cuda.current_stream().cuda_stream)==0
    launch()
    assert not state[:864].any()
    return launch,out
# Keep the own page first and the same complete selected set. Swap one anchor with an extra page.
ids[:,:,[1,25]]=ids[:,:,[25,1]]
base,ref=make('base')
out={}
for name in ('prefix','skip','words'):
    _,actual=make(name)
    delta=actual.float()-ref.float()
    out[name]=dict(changed=int((actual!=ref).sum()),nmse=float(delta.square().sum()/ref.float().square().sum()))
assert out['prefix']['changed'] and not out['skip']['changed'] and not out['words']['changed']
(root/f'order_{p}.json').write_text(json.dumps(out,indent=2))
