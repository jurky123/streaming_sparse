"""Keep both probability terms and use the persistent output as the MMA addend."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
src=(root.parent/'cuda_base_layout_v1/fixed15_attention.cu').read_text()
a=src.index('        #pragma unroll\n        for(int n=0;n<16;n+=2)',src.index('__global__ void exact'))
b=src.index('    }\n    __syncthreads();\n    float* E',a)
old=src[a:b]
for name in ('carry','carry_wide'):
    if name=='carry':
        av=old[:old.index('            #pragma unroll\n            for(int t=0;')].replace('Four tmp[2]={}; ','').replace('tmp[0]','out[n]').replace('tmp[1]','out[n+1]')+'        }\n'
    else:
        av='''
        #pragma unroll
        for(int term=0;term<2;term++)
            #pragma unroll
            for(int k=0;k<4;k++)
                #pragma unroll
                for(int n=0;n<16;n+=2){
                    uint4 u=kv[1][(k*8+n/2)*32+l];
                    A prob=term?low[k]:high[k];
                    mma(out[n],prob,B{{u.x,u.y}});mma(out[n+1],prob,B{{u.z,u.w}});
                }
'''
    out=root/(name+'.cu');out.write_text(src[:a]+av+src[b:])
    subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_80','--fmad=false','-shared','-Xcompiler=-fPIC',str(out),'-o','/tmp/jzk_'+name+'.so'],check=True)
