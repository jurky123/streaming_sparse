"""Skip a complete masked tile after issuing its successor's asynchronous copy."""
from pathlib import Path
import subprocess
s=Path('/tmp/jzk_compact_half.cu').read_text()
mark='        int pg=16*step;'
assert s.count(mark)==1
s=s.replace(mark,mark+'''
        unsigned selected=mask[pg/32]>>(pg%32);
        if((selected&0xffffu)==0xffffu)return;''')
s=s.replace('mask[pg/32]>>((pg%32)+2*z)','selected>>(2*z)')
out=Path('/tmp/jzk_full_mask.cu')
out.write_text(s)
subprocess.run(['nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o',str(out.with_suffix('.so'))],check=True)
