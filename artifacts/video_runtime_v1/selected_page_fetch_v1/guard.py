"""Compare the installed kernel and interleaved kernel in alternating CUDA graphs."""
import ctypes as C
import sys,json,os
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
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
from local_experiments.video_runtime_v1.fixed15_layout import retry
def check():
    old,a,af=implementation(C.CDLL('/tmp/jzk_route_control.so'))
    new,b,bf=implementation(library())
    assert torch.equal(af,bf)
    for out,flags in ((a,af),(b,bf)):
        retry[(72,12)](flags,q,k,v,ids,keys,mean,detail,basis,out,p,hp,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
    assert torch.equal(a,b) and torch.isfinite(a).all()
    return int(af.sum())

normal=check()
k[:,:,0].zero_();k[:,:64,0].fill_(10);q[:,:64,0].fill_(10)
pack_range[(p,12,16)](k,v,pk,pv,0)
partial=check()
k.zero_();k[:,:64].fill_(10);q.fill_(10)
pack_range[(p,12,16)](k,v,pk,pv,0)
extreme=check()
assert normal==0 and 0<partial<864 and extreme==864,(normal,partial,extreme)
(root/f'guard_{p}.json').write_text(json.dumps(dict(p=p,normal=normal,partial=partial,extreme=extreme,bitwise=True)))
