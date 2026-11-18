"""Fold probability columns into lane-local bins across pages, then reduce once."""
from pathlib import Path
import subprocess
d=Path(__file__).parent
s=(Path(__file__).parent/'control.cu').read_text()
for n in (2,4):
    name=f'mass{n}'
    src=s.replace('Four out[16]={},cols[8]={};','Four out[16]={};float masses[2]['+str(n)+']={};')
    src=src.replace('cols[n].x[r*2]+=x; cols[n].x[r*2+1]+=y;',f'masses[r][n%{n}]+=x+y;')
    a=src.index('        float partial[4];')
    b=src.index('        int idx=',a)
    src=src[:a]+f'''        float s=0;
        #pragma unroll
        for(int j=0;j<{n};j++)s+=masses[r][j];
        s+=__shfl_xor_sync(0xffffffff,s,2);
        s+=__shfl_xor_sync(0xffffffff,s,1);
'''+src[b:]
    f=d/(name+'.cu');f.write_text(src)
    subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-shared','-Xcompiler=-fPIC',str(f),'-o','/tmp/jzk_'+name+'.so'],check=True)
