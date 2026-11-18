"""Compile the deployed CUDA loop for the fixed P432 contract and mask boundaries."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
src=(root.parent/'cuda_base_layout_v1/fixed15_attention.cu').read_text()
src=src.replace('    extern __shared__ uint4 pool[];', '    p=432; count=64;\n    extern __shared__ uint4 pool[];')
src=src.replace('    for(int step=0;step<tiles;step++){','    #pragma unroll 1\n    for(int step=0;step<tiles;step++){').replace('    for(int step=-1;step<count;step++){','    #pragma unroll 1\n    for(int step=-1;step<count;step++){')
src=src.replace('page<p && !(bits&','!(bits&')
out=root/'constant.cu';out.write_text(src)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_80','-Xptxas=-v','-shared','-Xcompiler=-fPIC',str(out),'-o','/tmp/jzk_constant.so'],check=True)
