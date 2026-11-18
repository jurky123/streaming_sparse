"""Issue the previous output pair's AV low term alongside the next pair's high term."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('        #pragma unroll',src.index('// Accumulate both BF16'))
b=src.index('    }\n    __syncthreads();',a)
s=src[:a]+'''        uint4 v_pair[2][4];
        #pragma unroll
        for(int group=0;group<=8;group++){
            #pragma unroll
            for(int k=0;k<4;k++){
                if(group<8){
                    uint4 v=kv[1][(k*8+group)*32+l];
                    v_pair[group%2][k]=v;
                    mma(out[2*group],high[k],B{{v.x,v.y}});
                    mma(out[2*group+1],high[k],B{{v.z,v.w}});
                }
                if(group>0){
                    uint4 v=v_pair[(group-1)%2][k];
                    mma(out[2*group-2],low[k],B{{v.x,v.y}});
                    mma(out[2*group-1],low[k],B{{v.z,v.w}});
                }
            }
        }
'''+src[b:]
out=Path('/tmp/jzk_page_overlap.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
