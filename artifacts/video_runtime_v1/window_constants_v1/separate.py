"""Isolate specialization of the compensation tile count from that of the exact-page count."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
source=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
def build(spec):
    name,p=spec
    tail='int tiles=(p+15)/16;'
    assert source.count(tail)>=1
    if name=='tiles':
        src=source.replace(tail,f'constexpr int tiles={(p+15)//16};',1)
    else:
        target='int* bad,int p,int count) {'
        assert source.count(target)==1
        src=source.replace(target,f'int* bad,int p,int) {{\n    constexpr int count={p*15//100};')
    src=src.replace('    constexpr int bytes=',f'    if(p!={p} || count!={p*15//100})return cudaErrorInvalidValue;\n    constexpr int bytes=',1)
    path=Path(f'/tmp/jzk_route_{name}{p}.cu')
    path.write_text(src)
    subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
with ThreadPoolExecutor(4) as pool:
    list(pool.map(build,[(name,p) for name in ('tiles','pages') for p in (216,288,360,432)]))
