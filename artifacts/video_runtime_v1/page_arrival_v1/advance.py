"""Advance the next page's first copies past the previous last shared read."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('    for(int step=-1;')
b=src.index('        if(step+1<count)',a)
c=src.index('        if(step<0)continue;',b)
copy=src[b:c]
s=src[:a]+'    auto prefetch=[&](int step){\n'+copy+'''    };
    auto advance=[&](int step){
        asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
        prefetch(step);
    };
    prefetch(-1);
    advance(0);
    for(int step=0;step<count;step++){
        int pg=step;
'''+src[c+len('        if(step<0)continue;'):]
key='''            #pragma unroll
            for(int k=0;k<4;k++) {
                mma(out[n],low'''
assert s.count(key)==1
s=s.replace(key,'''            if(n==14 && step+1<count) advance(step+1);
'''+key)
out=Path('/tmp/jzk_page_advance.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
