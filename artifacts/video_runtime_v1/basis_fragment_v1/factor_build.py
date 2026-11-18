"""Control the tail loop bound independently of the physical per-head summary stride."""
from pathlib import Path
import subprocess
src=(Path(__file__).resolve().parents[3]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('__device__ __forceinline__ void tail(')
b=src.index('__global__ void exact(',a)
tail=src[a:b].replace('const unsigned* mask){','const unsigned* mask,int tiles){').replace('    int tiles=(p+15)/16;','').replace('(h*tiles+next)','(h*((p+15)/16)+next)')
src=src[:a]+tail+src[b:].replace('int p,int count) {','int p,int count,int tiles) {',1).replace('count,mask);','count,mask,tiles);',1).replace('int count,cudaStream_t stream){','int count,int tiles,cudaStream_t stream){',1).replace('basis,bad,p,count);','basis,bad,p,count,tiles);',1)
# P216 retains the physical 32-ID row when only its prefix is executed.
assert src.count('ids+(h*72+t)*count') == 1
src=src.replace('ids+(h*72+t)*count','ids+(h*72+t)*32')
out=Path('/tmp/jzk_basis_short_factor.cu');out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
