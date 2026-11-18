"""Use eight explicit rounds of both K and V for the P64 exact-page copy."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
src=(root.parents[2]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
key='            for(int m=threadIdx.x;m<1024;m+=128){'
assert src.count(key)==1
src=src.replace(key,'''            #pragma unroll
            for(int round=0;round<8;round++){
                int m=threadIdx.x+128*round;''')
out=Path('/tmp/jzk_slope_exact_copy.cu')
out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
