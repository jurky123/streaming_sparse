"""Seed the 24 guaranteed anchor bits and insert only own and extra pages."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
old='''if(step==0 && threadIdx.x<count){
            int id=list[threadIdx.x];
            atomicOr(mask+id/32,1u<<(id%32));
        }'''
# The initial 24 pages have no tail weights even when presented in a different order.
new='''if(step==0 && threadIdx.x<count){
            int id=list[threadIdx.x];
            if(id>=24)atomicOr(mask+id/32,1u<<(id%32));
        }'''
assert src.count(old)==1
src=src.replace(old,new)
path=Path('/tmp/jzk_route_skip.cu')
path.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
