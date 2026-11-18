"""Consume one ready-made bitset per route, appended after the unchanged ID table."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('    // Padded positions')
b=src.index('    const int* list=',a)
src=src[:a]+'''    if(threadIdx.x<16)
        mask[threadIdx.x]=ids[864*count+(h*72+t)*16+threadIdx.x];
'''+src[b:]
a=src.index('        // The first page barrier precedes')
b=src.index('        Slot& kv=',a)
src=src[:a]+src[b:]
path=Path('/tmp/jzk_route_prepared.cu')
path.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(path),'-o',str(path.with_suffix('.so'))],check=True)
