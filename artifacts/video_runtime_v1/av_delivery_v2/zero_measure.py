from pathlib import Path
import ctypes as C
import sys,json
import torch
from local_experiments.video_runtime_v1.fixed15_layout import encode
from local_experiments.video_runtime_v1.fixed15_attention import library
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,pk,pv,ids,summary,basis,p,hp,count,keys,mean,detail,implementation
tiles=(p+15)//16
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
base,expected,flags=implementation(library())
out=torch.empty_like(q)
c=torch.full((12*72*4,2),-1,device=q.device,dtype=torch.int32)
bad=torch.empty_like(flags)
lib=C.CDLL('/tmp/jzk_av_zero_count.so')
lib.run.argtypes=[C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]*2
assert lib.run(*(t.data_ptr() for t in (q,pk,pv,ids,out,summary,basis,bad)),p,count,c.data_ptr(),torch.cuda.current_stream().cuda_stream)==0
assert not bad.any() and torch.equal(expected.view(torch.uint8),out.view(torch.uint8)) and (c>=0).all()
high,low=c.sum(0).tolist()
row=dict(p=p,fragments_per_term=c.shape[0]*4*count,zero_high=high,zero_low=low,bitwise=True)
(root/f'zeros_{p}.json').write_text(json.dumps(row,indent=2));print(row)
