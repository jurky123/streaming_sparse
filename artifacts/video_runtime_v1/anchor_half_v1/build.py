"""Specialize the first tail tile to the eight pages after the permanent anchor."""
import hashlib
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
source=root/'control.cu'
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(source),'-o','/tmp/jzk_anchor_control.so'],check=True)
s=source.read_text()
a=s.index('    for(int step=1;step<tiles;step++){')
b=s.index('    // The opposite summary slot',a)
loop=s[a:b]
first=loop.replace('for(int step=1;step<tiles;step++){','{\n        constexpr int step=1;')
first=first.replace('for(int n=0;n<8;n+=2)','for(int n=4;n<8;n+=2)')
first=first.replace('float v=(lane_mask &','float v=n<4 || (lane_mask &')
s=s[:a]+first+loop.replace('step=1;','step=2;')+s[b:]
out=Path('/tmp/jzk_anchor_half.cu')
out.write_text(s)
print('base sha256',hashlib.sha256(source.read_bytes()).hexdigest(),flush=True)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC','-Xptxas=-v',str(out),'-o',str(out.with_suffix('.so'))],check=True)
