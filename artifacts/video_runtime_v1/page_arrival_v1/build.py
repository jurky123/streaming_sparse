"""Split the exact page barrier across the last eight low-term AV instructions."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
start=src.index('__global__ void exact(')
s=src[start:]
s=s.replace('    using Slot=','''    __shared__ unsigned long long page_barrier;
    unsigned bar=__cvta_generic_to_shared(&page_barrier);
    if(threadIdx.x==0) asm volatile("mbarrier.init.shared.b64 [%0], 128;"::"r"(bar):"memory");
    __syncthreads();
    unsigned long long token=0;
    auto arrive=[&](){
        asm volatile("cp.async.wait_group 0;":::"memory");
        asm volatile("mbarrier.arrive.shared.b64 %0,[%1];":"=l"(token):"r"(bar):"memory");
    };
    using Slot=''')
old='''            asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();'''
assert s.count(old)==1
s=s.replace(old,'''            unsigned ready;
            do {
                asm volatile("{ .reg .pred done; mbarrier.test_wait.shared.b64 done,[%1],%2; selp.u32 %0,1,0,done; }"
                             :"=r"(ready):"r"(bar),"l"(token):"memory");
            } while(!ready);''')
s=s.replace('        if(step<0)continue;', '        if(step<0){arrive();continue;}')
key='''            #pragma unroll
            for(int k=0;k<4;k++) {
                mma(out[n],low'''
assert s.count(key)==1
s=s.replace(key,'''            if(n==14 && step+1<count)arrive();
'''+key)
out=Path('/tmp/jzk_page_arrival.cu')
out.write_text(src[:start]+s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
