"""Seed the 24 guaranteed anchor bits and insert only own and extra pages."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
old='mask[threadIdx.x]=(threadIdx.x==p/32 && p%32)?(~0u<<(p%32)):0u;'
assert src.count(old)==1
src=src.replace(old,'mask[threadIdx.x]=threadIdx.x==0 ? 0x00ffffffu : ((threadIdx.x==p/32 && p%32)?(~0u<<(p%32)):0u);')
old='if(step==0 && threadIdx.x<count)'
assert src.count(old)==1
src=src.replace(old,'if(step==0 && threadIdx.x<count && (threadIdx.x==0 || threadIdx.x>=25))')
path=Path('/tmp/jzk_route_prefix.cu')
path.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
