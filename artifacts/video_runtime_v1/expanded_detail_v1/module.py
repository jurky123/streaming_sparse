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

old,a,fa,f,old_full=setup(library(),1328)
new,b,fb,g,new_full=setup(ctypes.CDLL('/tmp/jzk_expanded_detail.so'),1408)
assert torch.equal(old[:,:,:10240],new[:,:,:10240]) and not fa.any() and torch.equal(fa,fb)
# Check the packed low-rank operands against the logical four-representative sum.
i=torch.arange(512,device=q.device)
lane=(i//4)%32
pos=16*(i//128)+2*(lane%4)+8*((i//2)%2)+i%2
page=16*torch.arange(new.shape[1],device=q.device)[:,None]+8*(pos//32)+2*((pos%8)//2)+(pos%32)//16
rep=2*((pos//8)%2)+pos%2
components=detail[:,page.clamp(max=p-1),:,lane//4].permute(2,0,1,3).float()
signs=q.new_tensor([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]).float()
ref=(components*signs[rep]).sum(-1).masked_fill(page[None]>=p,0)
assert torch.equal(new[:,:,10240:10752],ref.bfloat16())
assert torch.equal(new[:,:,10752:],(ref-ref.bfloat16().float()).bfloat16())
delta=b.float()-a.float()
assert torch.isfinite(delta).all()
x,y,u,vv=times([timed(f),timed(g),timed(old_full),timed(new_full)])
def result(x,y):
    assert min(x+y)>0.1
    return dict(base_ms=median(x),new_ms=median(y),speedup=median(i/j for i,j in zip(x,y)),samples=[x,y])
out=dict(p=p,main=result(x,y),with_prepare=result(u,vv),max_abs=delta.abs().max().item(),
         nmse=(delta.square().sum()/a.float().square().sum()).item(),changed=int((a!=b).sum()))
(root/f'p{p}.json').write_text(json.dumps(out,indent=2))
print(out['main']['speedup'],out['with_prepare']['speedup'],out['nmse'])
