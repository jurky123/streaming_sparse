"""Stage only the live half of each K slice of the first compensation tile."""
from pathlib import Path
import subprocess
s=Path('/tmp/jzk_compact_half.cu').read_text()
a=s.index('            for(int i=threadIdx.x;i<words;i+=128)',s.index('On the last iteration exact'))
b=s.index('\n            }',a)+len('\n            }')
s=s[:a]+'''            #pragma unroll
            for(int round=0;round<4;round++){
                int i=256*round + 64 + threadIdx.x + 64*(threadIdx.x/64);
                unsigned addr=__cvta_generic_to_shared(pool+i);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(first+i));
            }
            for(int i=1024+threadIdx.x;i<words;i+=128){
                unsigned addr=__cvta_generic_to_shared(pool+i);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(first+i));
            }'''+s[b:]
out=Path('/tmp/jzk_half_copy.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
