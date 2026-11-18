"""Separate the shape constants from the explicit request to keep loops rolled."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
source=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
def build(name):
    if name=='rolled':
        src=source
        for loop in ['for(int step=0;step<count;step++){','for(int step=2;step<tiles;step++)']:
            src=src.replace(loop,'#pragma unroll 1\n    '+loop)
    else:
        p=int(name[4:])
        src=Path(f'/tmp/jzk_route_const{p}.cu').read_text()
        assert src.count('#pragma unroll 1\n')==2
        src=src.replace('#pragma unroll 1\n    ','')
    path=Path(f'/tmp/jzk_route_{name}.cu')
    path.write_text(src)
    subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
with ThreadPoolExecutor(3) as pool:
    list(pool.map(build,['rolled','auto216','auto432']))
