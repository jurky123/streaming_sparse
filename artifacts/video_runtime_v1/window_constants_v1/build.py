"""Specialize one CUDA kernel per deployed shape, retaining a rolled page loop."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
source=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
def build(p):
    count=p*15//100
    sig='int* bad,int p,int count) {'
    assert source.count(sig)==1
    src=source.replace(sig,f'int* bad,int,int) {{\n    constexpr int p={p},count={count};')
    for loop in ['for(int step=0;step<count;step++){','for(int step=2;step<tiles;step++)']:
        assert src.count(loop)==1
        src=src.replace(loop,'#pragma unroll 1\n    '+loop)
    src=src.replace('    constexpr int bytes=',f'    if(p!={p} || count!={count})return cudaErrorInvalidValue;\n    constexpr int bytes=',1)
    path=Path(f'/tmp/jzk_route_const{p}.cu')
    path.write_text(src)
    subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
with ThreadPoolExecutor(4) as pool:
    list(pool.map(build,[216,288,360,432]))
