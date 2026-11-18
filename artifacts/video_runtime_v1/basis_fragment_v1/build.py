"""Supply the rank-eight projection with two ready-to-issue TF32 fragments per lane."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
source = (root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a = source.index('        B b={{tf(__bfloat162float(basis[')
b = source.index('        C proj={}', a)
source = source[:a] + '''        uint4 word=reinterpret_cast<const uint4*>(basis)[(h*8+n/2)*32+l];
        B b=(n&1)?B{{word.z,word.w}}:B{{word.x,word.y}};
''' + source[b:]
out = Path('/tmp/jzk_page_basisfragment.cu')
out.write_text(source)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',
                str(out),'-o',str(out.with_suffix('.so'))],check=True)
