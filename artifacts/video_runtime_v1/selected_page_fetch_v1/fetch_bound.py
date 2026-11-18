"""Measure selected-page fetch cost using identical repeated-page arithmetic."""
from pathlib import Path
import ctypes as C,json,subprocess,sys
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
root=Path(__file__).parent
src=(Path(__file__).parent/'control.cu').read_text()
i=src.index('__global__ void exact(')
head,s=src[:i],src[i:]
same=s.replace('int page=list[step+1];','int page=list[0];',1)
reuse=same.replace('if(step>=0) {','if(step==0) {',1).replace('if(step+1<count){','if(step==-1){',1)
a=reuse.index('        }\n        asm volatile("cp.async.commit_group;')
reuse=reuse[:a]+reuse[a:].replace('        }\n        asm volatile("cp.async.commit_group;":::"memory");','            asm volatile("cp.async.commit_group;":::"memory");\n        }',1)
reuse=reuse.replace('Slot& kv=slots[step&1];','Slot& kv=slots[0];')
# Preserve all QK and dual AV instructions in both diagnostics.
libs=[]
for name,code in [('repeated_copy',same),('repeated_shared',reuse)]:
    f=root/(name+'.cu');f.write_text(head+code);binary='/tmp/jzk_'+name+'.so'
    subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-shared','-Xcompiler=-fPIC',str(f),'-o',binary],check=True)
    libs.append(C.CDLL(binary))
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import q,k,v,ids,basis,p,hp,count,keys,mean,detail,timed,times
from local_experiments.video_runtime_v1.fixed15_layout import encode,pack_range
pk,pv=torch.empty_like(k),torch.empty_like(v)
pack_range[(p,12,16)](k,v,pk,pv,0)
tiles=(p+15)//16
summary=q.new_empty((12,tiles,10624))
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,num_warps=4)
def make(lib):
    out=torch.empty_like(q);flags=torch.empty((12,72),device=q.device,dtype=torch.int32)
    lib.run.argtypes=[C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]
    def launch():
        assert lib.run(*(t.data_ptr() for t in (q,pk,pv,ids,out,summary,basis,flags)),p,count,torch.cuda.current_stream().cuda_stream)==0
    launch();assert not flags.any()
    return launch,out
runs=[make(lib) for lib in [C.CDLL('/tmp/jzk_route_control.so'),*libs]]
assert torch.equal(runs[1][1],runs[2][1])
result=times([timed(f) for f,_ in runs])
r=dict(p=p,ms=list(map(median,result)),labels=['deployed','repeated_copy','repeated_shared'],diagnostic_pair_bitwise=True,samples=result)
(root/f'fetch_bound_{p}.json').write_text(json.dumps(r,indent=2));print(r['ms'])
