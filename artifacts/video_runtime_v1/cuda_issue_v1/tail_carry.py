"""Evaluate the same two-term update for compensation with and without exact carry."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
for name,base in [('carry_tail',(root.parent/'cuda_base_layout_v1/fixed15_attention.cu')),
                  ('carry_both',root/'carry.cu')]:
    s=base.read_text()
    a=s.index('            C c={};mma(c,hi[0],b);')
    b=s.index('        }',a)
    s=s[:a]+'''            #pragma unroll
            for(int j=0;j<4;j++)am[n].v[j]*=fade[j/2];
            mma(am[n],hi[0],b);mma(am[n],lo[0],b);
'''+s[b:]
    out=root/(name+'.cu');out.write_text(s)
    subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_80','-shared','-Xcompiler=-fPIC',str(out),'-o','/tmp/jzk_'+name+'.so'],check=True)
