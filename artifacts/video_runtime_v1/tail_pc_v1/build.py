"""Emit the ten full-warp copy rounds and the 48-word remainder explicitly."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
src=(root.parents[2]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
start=src.index('            for(int i=threadIdx.x;i<words;i+=128)')
end=src.index('\n            }',start)+len('\n            }')
src=src[:start]+'''            unsigned dst=__cvta_generic_to_shared(slab[next%2]+threadIdx.x);
            const uint4* lane_input=input+threadIdx.x;
            #pragma unroll
            for(int round=0;round<words/128;round++){
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst+round*2048),"l"(lane_input+round*128));
            }
            if(threadIdx.x<words%128){
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst+20480),"l"(lane_input+1280));
            }'''+src[end:]
out=Path('/tmp/jzk_slope_copy_rounds.cu')
out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
