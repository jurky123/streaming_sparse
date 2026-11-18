"""Keep the winning slot parity and peeled tail, but issue tile 1 only after the guard."""
from pathlib import Path
import subprocess
s=Path('/tmp/jzk_slope_last_page.cu').read_text()
a=s.index('        else{\n            // On the last iteration')
b=s.index('        asm volatile("cp.async.commit_group;',a)
s=s[:a]+s[b:]
key='    for(int step=1;step<tiles;step++){'
assert s.count(key)==1
s=s.replace(key,'''    const uint4* first=reinterpret_cast<const uint4*>(KEY)+(h*tiles+1)*words;
    for(int i=threadIdx.x;i<words;i+=128){
        unsigned addr=__cvta_generic_to_shared(sm+i);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(first+i));
    }
    asm volatile("cp.async.commit_group;":::"memory");
'''+key)
out=Path('/tmp/jzk_slope_late_control.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
