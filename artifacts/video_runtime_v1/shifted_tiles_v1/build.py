"""Specialization for P216/P360: tile indices address original page 8 + 16*i."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
s=(root.parents[2]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
s=s.replace('(p+15)/16','(p-8)/16').replace('(pages+15)/16','(pages-8)/16')
s=s.replace('compute_tile(1,std::true_type{});','').replace('for(int step=2;step<tiles','for(int step=1;step<tiles')
s=s.replace('mask[threadIdx.x]=(threadIdx.x==p/32 && p%32)?(~0u<<(p%32)):0u;', 'mask[threadIdx.x]=0;')
s=s.replace('int id=list[threadIdx.x];\n            atomicOr','int id=list[threadIdx.x]-8;\n            if(id>=16)atomicOr')
s=s.replace('page=tile*16+','page=8+tile*16+').replace('token=tile*64+','token=32+tile*64+')
out=Path('/tmp/jzk_shifted_tiles.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
