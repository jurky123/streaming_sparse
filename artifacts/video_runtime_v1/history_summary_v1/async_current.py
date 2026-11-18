"""Copy current K/V to shared directly before warp fragment packing."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
s=Path('/tmp/jzk_compact_half.cu').read_text()
a=s.index('        *reinterpret_cast<uint4*>(&keys[row][col*8])=k[src];')
b=s.index('    __syncthreads();',a)
s=s[:a]+'''        unsigned ka=__cvta_generic_to_shared(&keys[row][col*8]);
        unsigned va=__cvta_generic_to_shared(&values[row][col*8]);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ka),"l"(k+src));
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(va),"l"(v+src));
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");
'''+s[b:]
out=Path('/tmp/jzk_async_current.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
