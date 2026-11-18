"""Specialize only exact page addressing for fixed-capacity, head-major packed K/V."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
s=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
for i in ('l','l+32'):
    old=f'(list[{i}]*12+h)*1024'
    assert s.count(old)==1
    s=s.replace(old,f'(h*432+list[{i}])*1024')
out=Path('/tmp/jzk_visit_head.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
