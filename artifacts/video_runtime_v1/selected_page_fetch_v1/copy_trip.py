"""Encode the deployed block's exact trip counts in global-to-shared copies."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
baseline=(Path(__file__).parent/'control.cu').read_text()
for name,s in [('fixed_copy',baseline),('route_copy',(root/'early_mask.cu').read_text())]:
    i=s.index('__global__ void exact(')
    head,kernel=s[:i],s[i:]
    kernel=kernel.replace('            for(int m=threadIdx.x;m<1024;m+=128){','''            #pragma unroll
            for(int part=0;part<8;part++){
                int m=threadIdx.x+part*128;''',1)
    head=head.replace('            for(int i=threadIdx.x;i<words;i+=128){','''            #pragma unroll
            for(int part=0;part<11;part++){
                int i=threadIdx.x+part*128;
                if(part==10 && threadIdx.x>=48)break;''',1)
    f=root/(name+'.cu');f.write_text(head+kernel)
    subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-lineinfo','-shared','-Xcompiler=-fPIC',str(f),'-o','/tmp/jzk_'+name+'.so'],check=True)
