"""Mark each CTA's arrival and completion with the device-wide timer and SM id."""
from pathlib import Path
import subprocess
src=(Path(__file__).resolve().parents[3]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('__global__ void exact(')
b=src.index('    extern __shared__',a)
src=src[:a]+src[a:b].replace('int count) {','int count,unsigned long long* stamps) {')+'''    auto* mark=stamps+(blockIdx.y*72+blockIdx.x)*3;
    if(threadIdx.x==0){
        unsigned sm; unsigned long long tm;
        asm volatile("mov.u32 %0,%%smid;":"=r"(sm));
        asm volatile("mov.u64 %0,%%globaltimer;":"=l"(tm));
        mark[0]=sm; mark[1]=tm;
    }
'''+src[b:]
key='    else asm volatile("cp.async.wait_group 0;":::"memory");'
assert src.count(key)==1
src=src.replace(key,key+'''
    __syncthreads();
    if(threadIdx.x==0){
        unsigned long long tm;
        asm volatile("mov.u64 %0,%%globaltimer;":"=l"(tm));
        mark[2]=tm;
    }''').replace('int count,cudaStream_t stream){','int count,unsigned long long* stamps,cudaStream_t stream){',1).replace('basis,bad,p,count);','basis,bad,p,count,stamps);',1)
out=Path('/tmp/jzk_issue_stamps.cu');out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
