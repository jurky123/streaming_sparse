"""Place each warp's FP32 exact-output fragments contiguously in shared memory."""
import subprocess
from pathlib import Path

source = Path(__file__).resolve().parents[3] / 'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu'
s = source.read_text()
for original in ('E[idx*128+n*8+pair*2+j]', 'exact[local*128+dim+j]'):
    assert s.count(original) == 1
    s = s.replace(original,original.split('[')[0]+'[((2*n+r)*128+threadIdx.x)*2+j]')
cu = Path('/tmp/jzk_exact_exchange.cu')
cu.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(cu),'-o',str(cu.with_suffix('.so'))],check=True)
