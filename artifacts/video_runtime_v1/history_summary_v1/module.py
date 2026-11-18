from pathlib import Path
from statistics import median
import ctypes
import json
import os
import sys
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import k,v,keys,mean,detail,hp,q,pk,pv,ids,basis,p,count,timed,times

tiles=(p+15)//16
only_kv=bool(os.environ.get("ONLY_KV"))
first=tiles if only_kv else (p-72)//16
lib=library()
partial=ctypes.CDLL('/tmp/jzk_only_current_kv.so' if only_kv else '/tmp/jzk_history_summary.so').prepare_fixed15
if os.getenv('DEPLOYED'):
    partial=lib.prepare_current_fixed15
partial.argtypes=lib.prepare_fixed15.argtypes
def setup(fn):
    summary=q.new_empty((12,tiles,10624))
    out=torch.empty_like(q)
    flags=torch.empty((12,72),dtype=torch.int32,device=q.device)
    args=[x.data_ptr() for x in (k,v,pk,pv,keys,mean,detail,summary)]
    run_args=[x.data_ptr() for x in (q,pk,pv,ids,out,summary,basis,flags)]
    def prepare():
        assert fn(*args,p,hp,torch.cuda.current_stream().cuda_stream)==0
    def full():
        prepare()
        assert lib.run(*run_args,p,count,torch.cuda.current_stream().cuda_stream)==0
    return summary,out,flags,prepare,full
a,x,af,old,f=setup(lib.prepare_fixed15)
b,y,bf,new,g=setup(partial)
f()
b[:,:first].copy_(a[:,:first])
b[:,first:].fill_(float('nan'))
g()
assert torch.equal(a,b) and torch.equal(x,y) and not af.any() and torch.equal(af,bf)
samples=times([timed(old),timed(new),timed(f),timed(g)])
def pair(a,b):
    return dict(base_ms=median(a),new_ms=median(b),paired_us=1000*median(i-j for i,j in zip(a,b)),speedup=median(i/j for i,j in zip(a,b)),samples=[a,b])
result=dict(p=p,cached_tiles=first,tiles=tiles,cached_bytes=first*12*10624*2,bitwise=True,
            prepare=pair(*samples[:2]),complete=pair(*samples[2:]))
(root/f'{"deployed_" if os.getenv("DEPLOYED") else "kv_only_" if only_kv else ""}p{p}.json').write_text(json.dumps(result,indent=2)+'\n')
