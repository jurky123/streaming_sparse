"""Use four independent maxima before the tail's two shuffle reductions."""
from pathlib import Path
import subprocess
src=(Path(__file__).resolve().parents[3]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
a=src.index('            float nm=mx[r];');b=src.index('            nm=fmaxf(nm,__shfl_',a)
s=src[a:b].replace('float nm=mx[r];','float peak[4]={mx[r],-CUDART_INF_F,-CUDART_INF_F,-CUDART_INF_F};')
s=s.replace('nm=fmaxf(nm,v);','peak[n%4]=fmaxf(peak[n%4],v);')
assert 'peak[n%4]=' in s
s+='            float nm=fmaxf(fmaxf(peak[0],peak[1]),fmaxf(peak[2],peak[3]));\n'
out=Path('/tmp/jzk_page_maxtree.cu')
out.write_text(src[:a]+s+src[b:])
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
