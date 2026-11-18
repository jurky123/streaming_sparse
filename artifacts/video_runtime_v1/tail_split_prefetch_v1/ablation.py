import ctypes
import json
from pathlib import Path
from statistics import median
import os
import sys
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,pk,pv,ids,summary,basis,p,count,timed,times

def implementation(lib, keys=pk, values=pv):
    lib.run.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*2+[ctypes.c_void_p]
    output=torch.empty_like(q)
    flags=torch.empty((12,72),device=q.device,dtype=torch.int32)
    tensors=(q,keys,values,ids,output,summary,basis,flags)
    def invoke():
        assert lib.run(*(x.data_ptr() for x in tensors),p,count,torch.cuda.current_stream().cuda_stream)==0
    invoke()
    return invoke,output,flags

name=os.environ['CANDIDATE']
from capture import keys,mean,detail,hp
from local_experiments.video_runtime_v1.fixed15_layout import encode
nt=(p+15)//16
encode[(12,nt,42)](keys,mean,detail,summary,p,hp,nt,DIRECT_MASS=True,num_warps=4)
f,a,fa=implementation(ctypes.CDLL('/tmp/jzk_slope_late_control.so'))
g,b,fb=implementation(ctypes.CDLL('/tmp/jzk_slope_'+name+'.so'))
assert torch.equal(fa,fb) and not fa.any()
delta=b.float()-a.float()
bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8))
assert bitwise, (name,p,int((a!=b).sum()),float(delta.abs().max()))
if os.environ.get('COUNTERS'):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    f(); g()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    sys.exit()
x,y=times([timed(f),timed(g)])
out=dict(p=p,bitwise=bitwise,max_abs=float(delta.abs().max()),nmse=float(delta.square().sum()/a.float().square().sum()),
         base_ms=median(x),new_ms=median(y),speedup=median(a/b for a,b in zip(x,y)),samples=[x,y])
(root.parent/'tail_split_prefetch_v1'/f"late_vs_{name}{os.getenv('RUN_SUFFIX','')}_{p}.json").write_text(json.dumps(out,indent=2))
print({k:v for k,v in out.items() if k!='samples'})
