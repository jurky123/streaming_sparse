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
old,a,af=implementation(C.CDLL('/tmp/jzk_route_control.so'))
name=os.environ.get('CANDIDATE','issue')
new,b,bf=implementation(library() if name=='installed' else C.CDLL('/tmp/jzk_'+name+'.so'))
assert not af.any() and torch.equal(af,bf)
bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8))
delta=b.float()-a.float()
nmse=float(delta.square().sum()/a.float().square().sum())
assert bitwise or (name.startswith("mass") and torch.isfinite(delta).all())
if os.getenv('COUNTERS'):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    old();new()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    sys.exit()
x,y=times([timed(old),timed(new)])
result=dict(p=p,bitwise=bitwise,nmse=nmse,max_abs=float(delta.abs().max()),changed=int((a!=b).sum()),base_ms=median(x),candidate_ms=median(y),paired_speedup=median(i/j for i,j in zip(x,y)),samples=[x,y])
(root/f'{name}_{p}.json').write_text(json.dumps(result,indent=2))
print(result['paired_speedup'])
