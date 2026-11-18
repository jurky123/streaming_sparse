"""Use SM cycles from the same CTA for exact, transition, compensation and merge."""
from pathlib import Path
import subprocess,sys,ctypes as C,json
from statistics import median
import torch
root=Path(__file__).parent
src=(root.parent/'cuda_base_layout_v1/fixed15_attention.cu').read_text()
i=src.index('__global__ void exact(')
j=src.index('    extern __shared__',i)
src=src[:i]+src[i:j].replace('int p,int count)','int p,int count,unsigned long long* ticks)')+src[j:]
# Each marker records the first warp's arrival; the same thread records every timestamp.
src=src.replace('    Four out[16]={},cols[8]={};','    unsigned long long* stamp=ticks+(blockIdx.y*72+blockIdx.x)*5;\n    if(!threadIdx.x)stamp[0]=clock64();\n    Four out[16]={},cols[8]={};')
src=src.replace('    float* E=reinterpret_cast<float*>(pool+2656);','    if(!threadIdx.x)stamp[1]=clock64();\n    float* E=reinterpret_cast<float*>(pool+2656);')
src=src.replace('    if(valid)tailops::tail','    if(!threadIdx.x)stamp[2]=clock64();\n    if(valid)tailops::tail').replace('pool,p,p,count);','pool,p,p,count,stamp);')
i=src.index('void tail(');j=src.index('{',i);src=src[:j-2]+src[j-2:j].replace(')',' ,unsigned long long* stamp)')+src[j:]
src=src.replace('    A high,low;', '    if(!threadIdx.x)stamp[3]=clock64();\n    A high,low;')
src=src.replace('pool,p,p,count,stamp);','pool,p,p,count,stamp);\n    if(!threadIdx.x)stamp[4]=clock64();')
src=src.replace('int count,cudaStream_t stream){','int count,unsigned long long* ticks,cudaStream_t stream){').replace('basis,bad,p,count);','basis,bad,p,count,ticks);')
so='/tmp/jzk_fixed15_clocks.so'
out=root/'clocks.cu';out.write_text(src)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',so],check=True)
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import q,k,v,ids,basis,p,hp,count,keys,mean,detail,timed,times
from local_experiments.video_runtime_v1.fixed15_layout import encode,pack_range
from local_experiments.video_runtime_v1.fixed15_attention import library
pk,pv=torch.empty_like(k),torch.empty_like(v)
pack_range[(p,12,16)](k,v,pk,pv,0)
t=(p+15)//16
summary=q.new_empty((12,t,10624))
encode[(12,t,42)](keys,mean,detail,summary,p,hp,t,num_warps=4)
actual=torch.empty_like(q);flags=torch.empty((12,72),device=q.device,dtype=torch.int32)
ticks=torch.empty((864,5),device=q.device,dtype=torch.int64)
lib=C.CDLL(so);lib.run.argtypes=[C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]*2
base=library();expected=torch.empty_like(q)
def run(fn,out,*extra):
    assert fn(*(x.data_ptr() for x in (q,pk,pv,ids,out,summary,basis,flags)),p,count,*extra,torch.cuda.current_stream().cuda_stream)==0
for _ in range(40):run(lib.run,actual,ticks.data_ptr())
samples=[]
for _ in range(15):
    run(lib.run,actual,ticks.data_ptr())
    parts=ticks.diff(dim=1).cpu()
    assert parts.min()>0
    samples.append(parts.double().mean(0).tolist())
run(base.run,expected)
assert torch.equal(expected,actual) and not flags.any()
cycles=list(map(median,zip(*samples)))
bare,profiled=times([timed(lambda:run(base.run,expected)),
                     timed(lambda:run(lib.run,actual,ticks.data_ptr()))])
(root/f'phase_{p}.json').write_text(json.dumps(dict(wall_ms=[median(bare),median(profiled)],phase=['exact_pages','exact_finish','tail_tiles','project_merge'],mean_cycles_per_cta=cycles,share=[x/sum(cycles) for x in cycles],bitwise=True,samples=samples),indent=2))
