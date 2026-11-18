"""Issue each 64-bit B operand independently while preserving the dual AV chains."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('                uint4 u=',src.index('// Accumulate both BF16'))
b=src.index('\n            }',a)
src=src[:a]+'''                unsigned addr=__cvta_generic_to_shared(&kv[1][(k*8+n/2)*32+l]);
                asm volatile("ld.shared.v2.b32 {%0,%1},[%2];"
                             :"=r"(b[k][0].x[0]),"=r"(b[k][0].x[1]):"r"(addr));
                mma(out[n],high[k],b[k][0]);
                asm volatile("ld.shared.v2.b32 {%0,%1},[%2+8];"
                             :"=r"(b[k][1].x[0]),"=r"(b[k][1].x[1]):"r"(addr));
                mma(out[n+1],high[k],b[k][1]);'''+src[b:]
out=Path('/tmp/jzk_av_halfload.cu')
if '--volatile' in sys.argv:
    src=src.replace('ld.shared.v2.b32','ld.volatile.shared.v2.b32')
    out=out.with_name('jzk_av_volatile.cu')
out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
