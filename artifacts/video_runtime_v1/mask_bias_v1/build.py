"""Fold the known page exclusion bias into the representative's logit scaling."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3]
src=(root/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
key='''                    float v=(first && n<4) || (lane_mask & (1u<<(8*(n/4)+(n%4)/2))) ? -CUDART_INF_F
                        : __fmul_rn(score[n].v[r*2+j],0.12751743082459868f);'''
assert src.count(key)==1
src=src.replace(key,'''                    float bias=(lane_mask & (1u<<(8*(n/4)+(n%4)/2))) ? -CUDART_INF_F : 0.f;
                    float v=first && n<4 ? -CUDART_INF_F :
                        __fmaf_rn(score[n].v[r*2+j],0.12751743082459868f,bias);''')
out=Path('/tmp/jzk_page_maskbias.cu');out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
