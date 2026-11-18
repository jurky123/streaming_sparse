"""Allow the next iteration to consume K while V's async copy is outstanding."""
from pathlib import Path
import subprocess
d=Path(__file__).parent
s=(Path(__file__).parent/'control.cu').read_text()
i=s.index('__global__ void exact')
a,s=s[:i],s[i:]
s=s.replace('cp.async.wait_group 0;','cp.async.wait_group 1;',1)
i=s.index('            for(int m=threadIdx.x')
j=s.index('        if(step<0)',i)
s=s[:i]+'''            for(int m=threadIdx.x;m<1024;m+=128){
                unsigned addr=__cvta_generic_to_shared(&slots[slot][0][m]);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(fk+m));
            }
            asm volatile("cp.async.commit_group;":::"memory");
            for(int m=threadIdx.x;m<1024;m+=128){
                unsigned addr=__cvta_generic_to_shared(&slots[slot][1][m]);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(fv+m));
            }
        }else{
            asm volatile("cp.async.commit_group;":::"memory");
        }
        asm volatile("cp.async.commit_group;":::"memory");
'''+s[j:]
i=s.index('        // Accumulate both BF16')
s=s[:i]+'''        asm volatile("cp.async.wait_group 2;":::"memory");
        __syncthreads();
'''+s[i:]
f=d/'split_wait.cu';f.write_text(a+s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-lineinfo','-shared','-Xcompiler=-fPIC',str(f),'-o','/tmp/jzk_split_wait.so'],check=True)
