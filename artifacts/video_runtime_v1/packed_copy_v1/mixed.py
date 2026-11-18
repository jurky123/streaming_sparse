"""Use native K/V just for current pages, packed K/V for immutable history."""
from pathlib import Path
import subprocess
d=Path(__file__).parent
src=(d.parent/'cuda_base_layout_v1/fixed15_attention.cu').read_text()
raw=(d.parent/'fixed15_cuda_pipeline/rawboth.cu').read_text()
i=src.index('__global__ void exact')
head,s=src[:i],src[i:]
s=s.replace('int* bad,int p,int count) {','int* bad,int p,int count,const bf* native_k,const bf* native_v) {',1)
a=s.index('            for(int m=threadIdx.x')
b=s.index('\n        }\n        asm volatile',a)
s=s[:a]+'''            if(page>=p-72) {
                stage(reinterpret_cast<bf*>(slots[slot][0]),native_k,page,h);
                stage(reinterpret_cast<bf*>(slots[slot][1]),native_v,page,h);
            }else{\n'''+s[a:b]+'\n            }'+s[b:]
# Compile two instruction paths around each whole QK/AV operation, preserving the arithmetic.
a=s.index('        #pragma unroll',s.index('        Four score[8]={};'))
b=s.index('        #pragma unroll\n        for(int r=0;r<2;r++)',a)
ra=raw.index('        #pragma unroll',raw.index('        Four score[8]={};',raw.index('__global__ void exact')))
rb=raw.index('        #pragma unroll\n        for(int r=0;r<2;r++)',ra)
s=s[:a]+'        bool current=list[step]>=p-72;\n        if(current){\n'+raw[ra:rb]+'        }else{\n'+s[a:b]+'        }\n'+s[b:]
a=s.index('        #pragma unroll\n        for(int n=0;n<16;n+=2)',s.index('        A high[4]'))
b=s.index('    }\n    __syncthreads();\n    float* E',a)
ra=raw.index('        #pragma unroll\n        for(int n=0;n<16;n+=2)',raw.index('        A high[4]',raw.index('__global__ void exact')))
rb=raw.index('    }\n    __syncthreads();\n    float* E',ra)
s=s[:a]+'        if(current){\n'+raw[ra:rb]+'        }else{\n'+s[a:b]+'        }\n'+s[b:]
s=s.replace('int count,cudaStream_t stream){','int count,const bf* native_k,const bf* native_v,cudaStream_t stream){',1).replace('basis,bad,p,count);','basis,bad,p,count,native_k,native_v);')
f=d/'mixed.cu';f.write_text(head+s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_80','-Xptxas=-v','-shared','-Xcompiler=-fPIC',str(f),'-o','/tmp/jzk_mixed_current.so'],check=True)
