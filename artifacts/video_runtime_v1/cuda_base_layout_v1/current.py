"""Compare producing native and packed current K/V in the rotary K epilogue."""
import ctypes as C
import json, sys
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_layout import pack_range
from local_experiments.video_runtime_v1.rms_mean import row_rope
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import k,v,timed,times
x=k[:,-4608:].reshape(1,4608,1536)
value=v[:,-4608:]
weight=torch.ones(1536,device='cuda',dtype=x.dtype)
freq=torch.polar(torch.ones(256,64,device='cuda',dtype=torch.double), torch.randn(256,64,device='cuda',dtype=torch.double))
native,other,pk,pv,rk,rv=[torch.empty_like(value) for _ in range(6)]
def original():
    row_rope(x,weight,1e-6,freq,32,48,9,native)
    pack_range[(72,12,16)](native,value,rk,rv,0)
lib=C.CDLL('/tmp/jzk_emit_current.so')
lib.emit.argtypes=[C.c_void_p]*3+[C.c_int,C.c_float,C.c_void_p]+[C.c_int]*3+[C.c_void_p]*4
def direct():
    lib.emit(x.data_ptr(),weight.data_ptr(),other.data_ptr(),4608,1e-6,freq.data_ptr(),32,48,9,pk.data_ptr(),pv.data_ptr(),value.data_ptr(),torch.cuda.current_stream().cuda_stream)
original();direct()
assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in ((native,other),(pk,rk),(pv,rv)))
a,b=times([timed(original),timed(direct)])
row=dict(native_then_pack_ms=median(a),rope_emit_ms=median(b),bitwise=True)
(root/'current.json').write_text(json.dumps(row,indent=2))
print(row)
