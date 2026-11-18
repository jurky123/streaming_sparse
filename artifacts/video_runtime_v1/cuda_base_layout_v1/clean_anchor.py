"""Measure full clean attention with native and two MMA-packed consumers."""
import ctypes as C
import json,sys
from pathlib import Path
from statistics import median
import torch
from layout_trials import clean
from local_experiments.video_runtime_v1.fixed15_layout import pack_range
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import times,timed
inputs=torch.load('/tmp/jzk_cuda_clean_input.pt',weights_only=True)
q,k,v=(inputs[n].cuda().contiguous() for n in ('q','k','v'))
p=k.shape[1]//64
pk,pv=torch.empty_like(k),torch.empty_like(v)
out=torch.empty_like(q)
pack_range[(p,12,16)](k,v,pk,pv,0)
lib=C.CDLL('/tmp/jzk_clean_anchor.so')
flags=torch.empty((12,72),device='cuda',dtype=torch.int32)
lib.run.argtypes=[C.c_void_p]*4+[C.c_int,C.c_void_p,C.c_void_p]
def native():return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
def cuda():assert lib.run(*(t.data_ptr() for t in (q,pk,pv,out)),p,flags.data_ptr(),torch.cuda.current_stream().cuda_stream)==0
def triton():clean[(72,12)](q,pk,pv,out,p,num_stages=2,num_warps=4)
ref=native().float()
r=dict(chunk=inputs['chunk'],step=4,layer=1,pages=p)
for name,func in [('cuda',cuda)]:
    func()
    delta=ref-out.float()
    r[name+'_nmse']=float(delta.square().sum()/ref.square().sum())
    r[name+'_max']=float(delta.abs().max())
for name,seq in zip(('native','cuda'),times([timed(f) for f in (native,cuda)])):
    r[name+'_ms']=median(seq)
r['flagged']=int(flags.sum())
(root/'clean_anchor.json').write_text(json.dumps(r,indent=2))
print(r)
