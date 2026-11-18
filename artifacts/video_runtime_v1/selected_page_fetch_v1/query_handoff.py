"""Reuse exact Q MMA fragments as the compensation Q operand for the same warp rows."""
from pathlib import Path
import subprocess
d=Path(__file__).parent
s=(d/'early_mask.cu').read_text()
s=s.replace('void tail(const bf* Q,','void tail(const ::A (&query)[8],',1)
a=s.index('            q[k].v[j]=*reinterpret_cast')
b=s.index(';',a)
s=s[:a]+'            q[k].v[j]=query[k].x[j]'+s[b:]
s=s.replace('if(valid)tailops::tail(Q,','if(valid)tailops::tail(q,')
f=d/'query_handoff.cu';f.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-lineinfo','-Xptxas=-v','-shared','-Xcompiler=-fPIC',str(f),'-o','/tmp/jzk_query_handoff.so'],check=True)
