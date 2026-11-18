"""Two ordered async groups per compensation tile, K followed by mean/detail."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
s=(root.parent/'page_slope_v1/control.cu').read_text()
start=s.index('    for(int step=0;step<tiles;step++)')
end=s.index('        int pg=16*step;',start)
part=s[start:end]
copy=part[part.index('        if(step+1<tiles)'):part.index('        if(!step)continue')]
value_copy=copy.replace('i=threadIdx.x','i=1024+threadIdx.x')
part=part.replace('wait_group 0','wait_group 1').replace('i<words','i<1024').replace('        if(!step)continue;', '        if(!step){\n'+value_copy+'            continue;\n        }')
s=s[:start]+part+s[end:]
marker='        #pragma unroll\n        for(int n=0;n<16;n++){\n            uint4 raw=data'
assert s.count(marker)==1
s=s.replace(marker,'        asm volatile("cp.async.wait_group 1;":::"memory");\n        __syncthreads();\n'+value_copy+marker)
out=Path('/tmp/jzk_slope_tail_split.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
