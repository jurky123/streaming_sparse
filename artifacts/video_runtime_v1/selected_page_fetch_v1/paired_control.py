"""Separate the scheduling effect of a 256-thread CTA from shared prefix traffic."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
src=(root/'paired_cuda.cu').read_text()
a=src.index('__device__ __forceinline__ void tail(');b=src.index('extern "C" int run(',a)
head,k,end=src[:a],src[a:b],src[b:]
k=k.replace('bool same=step>=0 && step<24;', 'bool same=false;')
k=k.replace('(step>=1 && step<=24?reinterpret_cast<Slot*>(common_pool):slots)[step&1]','slots[step&1]')
k=k.replace('E,L,common_pool,p,p,count','E,L,pool,p,p,count').replace('for(int i=threadIdx.x;i<words;i+=256)','for(int i=tid;i<words;i+=128)')
for name in ('paired_ind','paired_named'):
    if name=='paired_named':
        k=k.replace('__syncthreads();','asm volatile("bar.sync %0, 128;"::"r"(group+1):"memory");')
    f=root/(name+'.cu');f.write_text(head+k+end)
    subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-shared','-Xcompiler=-fPIC',str(f),'-o','/tmp/jzk_'+name+'.so'],check=True)
