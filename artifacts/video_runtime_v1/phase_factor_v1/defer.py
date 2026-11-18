"""Keep tail mass per lane until the final tile: all four lanes have identical fade."""
from pathlib import Path
import subprocess
src=(Path(__file__).resolve().parents[3]/'local_experiments/video_runtime_v1/csrc/fixed15_attention.cu').read_text()
key='            partial+=__shfl_xor_sync(~0u,partial,2);partial+=__shfl_xor_sync(~0u,partial,1);\n'
assert src.count(key)==1
src=src.replace(key,'')
key='        float safe=fmaxf(den[r],1e-30f);'
assert src.count(key)==1
src=src.replace(key,'        den[r]+=__shfl_xor_sync(~0u,den[r],2);\n        den[r]+=__shfl_xor_sync(~0u,den[r],1);\n'+key)
out=Path('/tmp/jzk_defer_mass.cu');out.write_text(src)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
