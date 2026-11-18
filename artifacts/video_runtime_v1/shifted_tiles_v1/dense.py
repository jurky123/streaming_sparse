import ctypes
import json
from pathlib import Path
from statistics import median
import sys
import torch

from local_experiments.video_runtime_v1.fixed15_attention import library
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import k,v,keys,mean,detail,hp,q,pk,pv,ids,basis,p,count,timed,times

def setup(lib, words):
    packed=q.new_empty((12,(p+15)//16,8*words))
    out=torch.empty_like(q)
    flags=torch.empty((12,72),device=q.device,dtype=torch.int32)
    abi=[ctypes.c_void_p]*8+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.run.argtypes=lib.prepare_fixed15.argtypes=abi
    def prepare():
        args=(k,v,pk,pv,keys,mean,detail,packed)
        assert lib.prepare_fixed15(*(x.data_ptr() for x in args),p,hp,torch.cuda.current_stream().cuda_stream)==0
    def main():
        args=(q,pk,pv,ids,out,packed,basis,flags)
        assert lib.run(*(x.data_ptr() for x in args),p,count,torch.cuda.current_stream().cuda_stream)==0
    def full():
        prepare()
        main()
    full()
    return packed,out,flags,main,full

old,a,fa,f,old_full=setup(ctypes.CDLL('/tmp/jzk_compact_half.so'),1328)
new,b,fb,g,new_full=setup(ctypes.CDLL('/tmp/jzk_shifted_tiles.so'),1328)
assert p in (216,360) and not fa.any() and torch.equal(fa,fb)
# Independently pack reference operands after slicing away the first eight pages.
from local_experiments.video_runtime_v1.fixed15_layout import encode
nt=(p-8)//16
reference=q.new_empty((12,nt,10624))
encode[(12,nt,42)](keys[:,8:],mean[:,8:],detail[:,8:],reference,p-8,hp,nt,DIRECT_MASS=True,num_warps=4)
assert torch.equal(reference.flatten(),new.flatten()[:reference.numel()])
delta=b.float()-a.float()
assert torch.isfinite(delta).all()
def dense():
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
x,y,z=times([timed(f),timed(g),timed(dense)])
dense_ms=median(z)
dense_gflop=4*q.shape[1]*12*128*p*64/1e9
old_gflop=(192*count+102*((p+15)//16-1)-32)*3456*4096/1e9
new_gflop=(192*count+102*((p-24)//16))*3456*4096/1e9
row=dict(p=p,dense_ms=dense_ms,base_ms=median(x),candidate_ms=median(y),
         base_gflop=old_gflop,candidate_gflop=new_gflop,
         base_fraction=(old_gflop/median(x))/(dense_gflop/dense_ms),
         candidate_fraction=(new_gflop/median(y))/(dense_gflop/dense_ms),
         paired_speedup=median(a/b for a,b in zip(x,y)),samples=[x,y,z])
(root/f'dense_{p}.json').write_text(json.dumps(row,indent=2))
