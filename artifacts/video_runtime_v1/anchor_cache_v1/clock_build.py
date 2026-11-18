"""Use SM cycles from the same CTA for exact, transition, compensation and merge."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
source=root.parents[2]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu'
src=source.read_text()
i=src.index('__global__ void exact(')
j=src.index('    extern __shared__',i)
src=src[:i]+src[i:j].replace('int p,int count)','int p,int count,unsigned long long* ticks)')+src[j:]
# Each marker records the first warp's arrival; the same thread records every timestamp.
src=src.replace('    Four out[16]={},cols[8]={};','    unsigned long long* stamp=ticks+(blockIdx.y*72+blockIdx.x)*5;\n    if(!threadIdx.x)stamp[0]=clock64();\n    Four out[16]={},cols[8]={};')
src=src.replace('    float* E=reinterpret_cast<float*>(pool+2656);','    if(!threadIdx.x)stamp[1]=clock64();\n    float* E=reinterpret_cast<float*>(pool+2656);')
src=src.replace('    if(valid)tailops::tail','    if(!threadIdx.x)stamp[2]=clock64();\n    if(valid)tailops::tail').replace('pool,p,p,count,mask);','pool,p,p,count,mask,stamp);')
i=src.index('void tail(');j=src.index('{',i);src=src[:j-1]+src[j-1:j].replace(')',' ,unsigned long long* stamp)')+src[j:]
src=src.replace('    A high,low;', '    if(!threadIdx.x)stamp[3]=clock64();\n    A high,low;')
src=src.replace('else asm volatile("cp.async.wait_group 0;":::"memory");','else asm volatile("cp.async.wait_group 0;":::"memory");\n    if(!threadIdx.x)stamp[4]=clock64();')
src=src.replace('int count,cudaStream_t stream){','int count,unsigned long long* ticks,cudaStream_t stream){').replace('basis,bad,p,count);','basis,bad,p,count,ticks);')
so='/tmp/jzk_anchor_cache_clocks.so'
out=Path('/tmp/jzk_anchor_cache_clocks.cu');out.write_text(src)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',so],check=True)
