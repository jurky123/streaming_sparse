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
current,_,flags=implementation(library())
assert not flags.any()
def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
x,y=times([timed(dense),timed(current)])
warps=3456
dense_ops=4*q.shape[1]*12*128*p*64
sparse_ops=(count*192+((p+15)//16-1)*102)*warps*4096
dense_ms=median(x)
row=dict(p=p,count=count,dense_ms=dense_ms,fused_ms=median(y),dense_gflop=dense_ops/1e9,
         sparse_gflop=sparse_ops/1e9,dense_tflops=dense_ops/1e9/dense_ms,
         reference_ms=dense_ms*sparse_ops/dense_ops,samples=[x,y])
(root/f'dense_{p}.json').write_text(json.dumps(row,indent=2))
print(row['dense_tflops'])
