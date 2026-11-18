"""Set the exact softmax scale from the first eight keys of the own page."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
start=src.index('        if(pg==0) {')
stop=src.index('        A high[4]',start)
assert src[start:stop].count('n<8;')==1
src=src[:start]+src[start:stop].replace('n<8;','n<1;')+src[stop:]
out=Path('/tmp/jzk_issue_anchor8.cu')
out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
