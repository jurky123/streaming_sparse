import ctypes
import json
from pathlib import Path
from statistics import median
import os
import sys
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import k,v,keys,mean,detail,hp,q,pk,pv,ids,summary,basis,p,count,timed,times

def implementation(lib, keys=pk, values=pv, packed_summary=summary):
    lib.run.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*2+[ctypes.c_void_p]
    output=torch.empty_like(q)
    flags=torch.empty((12,72),device=q.device,dtype=torch.int32)
    tensors=(q,keys,values,ids,output,packed_summary,basis,flags)
    def invoke():
        assert lib.run(*(x.data_ptr() for x in tensors),p,count,torch.cuda.current_stream().cuda_stream)==0
    invoke()
    return invoke,output,flags

name=os.environ['CANDIDATE']
def setup(lib):
    packed=torch.empty_like(summary)
    lib.prepare_fixed15.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*2+[ctypes.c_void_p]
    def prepare():
        args=(k,v,pk,pv,keys,mean,detail,packed)
        assert lib.prepare_fixed15(*(x.data_ptr() for x in args),p,hp,torch.cuda.current_stream().cuda_stream)==0
    prepare()
    return prepare,implementation(lib,packed_summary=packed),packed
source='direct_mass' if name in ('release','pair_store','basis_bits','vector_store','vector_release') else 'control'
old_pack,(f,a,fa),original=setup(ctypes.CDLL('/tmp/jzk_tailmax_'+source+'.so'))
new_pack,(g,b,fb),new=setup(library() if name in ('release','vector_release') else ctypes.CDLL('/tmp/jzk_tailmax_'+name+'.so'))
if source=='direct_mass':
    assert torch.equal(original,new)
else:
    assert torch.equal(original,summary)
assert torch.equal(original[:,:,8192:],new[:,:,8192:])
assert torch.equal(fa,fb) and not fa.any()
delta=b.float()-a.float()
bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8))
assert bitwise or (name in ('center_fma','coeff_mass','direct_mass','direct_order') and torch.isfinite(delta).all())
if name=='direct_mass':
    _,(_,reference,rf),_=setup(ctypes.CDLL('/tmp/jzk_tailmax_coeff_mass.so'))
    assert torch.equal(reference.view(torch.uint8),b.view(torch.uint8)) and torch.equal(rf,fb)
if os.environ.get('COUNTERS'):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    f(); g()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    sys.exit()
x,y=times([timed(f),timed(g)])
out=dict(p=p,bitwise=bitwise,max_abs=float(delta.abs().max()),nmse=float(delta.square().sum()/a.float().square().sum()),
         base_ms=median(x),new_ms=median(y),speedup=median(a/b for a,b in zip(x,y)),samples=[x,y])
if name.startswith('direct') or name=='coeff_mass':
    def old_full():
        old_pack(); f()
    def new_full():
        new_pack(); g()
    u,vv=times([timed(old_full),timed(new_full)])
    out['with_prepare']=dict(old_ms=median(u),new_ms=median(vv),speedup=median(i/j for i,j in zip(u,vv)),samples=[u,vv])
(root/f"{name}{os.getenv('RUN_SUFFIX','')}_{p}.json").write_text(json.dumps(out,indent=2))
print(name,p,out['speedup'],out.get('with_prepare',{}).get('speedup'),out['nmse'])
