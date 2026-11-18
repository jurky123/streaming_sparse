"""Load the captured input and set up the packed SM80 kernel arguments."""
import ctypes as C
import sys
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.fixed15_layout import pack_range,encode

root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import q,k,v,ids,basis,p,hp,count,keys,mean,detail,timed,times
pk,pv=torch.empty_like(k),torch.empty_like(v)
pack_range[(p,12,16)](k,v,pk,pv,0)
tiles=(p+15)//16
summary=q.new_empty((12,tiles,10624))
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,num_warps=4)
def implementation(lib):
    lib.run.argtypes=[C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]
    out=torch.empty_like(q)
    flags=torch.empty((12,72),device=q.device,dtype=torch.int32)
    def invoke():
        assert lib.run(*(x.data_ptr() for x in (q,pk,pv,ids,out,summary,basis,flags)),p,count,torch.cuda.current_stream().cuda_stream)==0
    invoke()
    return invoke,out,flags
