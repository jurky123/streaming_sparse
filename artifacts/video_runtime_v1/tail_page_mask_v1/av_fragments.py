"""Interleave four exact AV output fragments while loading each V operand once."""
from pathlib import Path
import subprocess

root=Path(__file__).parent
source=Path('/tmp/jzk_tailmask_lane_fma.cu')
if not source.exists():
    subprocess.run(['python',str(root/'build.py'),'lane_fma'],check=True)
src=source.read_text()
a=src.index('        #pragma unroll\n        for(int n=0;n<16;n+=2)',src.index('// Accumulate both BF16 probability terms'))
b=src.index('    }\n    __syncthreads();',a)
src=src[:a]+'''        #pragma unroll
        for(int first=0;first<16;first+=4) {
            uint4 value[4][2];
            #pragma unroll
            for(int k=0;k<4;k++)
                #pragma unroll
                for(int pair=0;pair<2;pair++){
                    int n=first+2*pair;
                    uint4 v=kv[1][(k*8+n/2)*32+l];
                    value[k][pair]=v;
                    mma(out[n],high[k],B{{v.x,v.y}});
                    mma(out[n+1],high[k],B{{v.z,v.w}});
                }
            #pragma unroll
            for(int k=0;k<4;k++)
                #pragma unroll
                for(int pair=0;pair<2;pair++){
                    int n=first+2*pair;
                    uint4 v=value[k][pair];
                    mma(out[n],low[k],B{{v.x,v.y}});
                    mma(out[n+1],low[k],B{{v.z,v.w}});
                }
        }
'''+src[b:]
out=Path('/tmp/jzk_tailmask_four_av.cu')
out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
