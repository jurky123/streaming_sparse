"""Build a diagnostic launch of only the current-page K/V grid."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
s=Path('/tmp/jzk_compact_half.cu').read_text()
launch='72*48+12*((pages+15)/16)*5'
assert s.count(launch)==1
out=Path('/tmp/jzk_only_current_kv.cu')
out.write_text(s.replace(launch,'72*48'))
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
