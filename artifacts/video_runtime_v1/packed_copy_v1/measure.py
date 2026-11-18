import ctypes as C
import json,sys,os
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_layout import pack_range
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import k,v,p,timed,times
name=os.environ.get('COPY','copy')
lib=C.CDLL('/tmp/jzk_'+name+'_frag.so')
lib.run.argtypes=[C.c_void_p]*4+[C.c_int,C.c_void_p]
a,b,c,d=[torch.zeros_like(k) for _ in range(4)]
def control():pack_range[(72,12,16)](k,v,a,b,p-72)
def candidate():
    assert lib.run(k.data_ptr(),v.data_ptr(),c.data_ptr(),d.data_ptr(),p-72,torch.cuda.current_stream().cuda_stream)==0
control();candidate()
assert torch.equal(a,c) and torch.equal(b,d),((a!=c).sum(),(b!=d).sum())
x,y=times([timed(control),timed(candidate)])
r=dict(p=p,base_ms=median(x),new_ms=median(y),speedup=median(i/j for i,j in zip(x,y)),bitwise=True,samples=[x,y])
(root/f'{name}_{p}.json').write_text(json.dumps(r,indent=2));print(r['speedup'])
