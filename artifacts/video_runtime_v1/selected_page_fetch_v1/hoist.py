"""Convert route indices to copy offsets before the loop; prepare the tail mask during its prologue."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
s=(root/'lane_ids.cu').read_text()
s=s.replace('int first=list[l], second=l+32<count?list[l+32]:0;','int first=(list[l]*12+h)*1024, second=l+32<count?(list[l+32]*12+h)*1024:0;')
s=s.replace('int page=__shfl_sync','int offset=__shfl_sync').replace('reinterpret_cast<const uint4*>(K)+(page*12+h)*1024','reinterpret_cast<const uint4*>(K)+offset').replace('reinterpret_cast<const uint4*>(V)+(page*12+h)*1024','reinterpret_cast<const uint4*>(V)+offset')
for name in ('offset','early_mask'):
    if name=='early_mask':
        a=s.index('    __shared__ unsigned mask[16];');b=s.index('    A q[8];C am',a)
        s=s[:a]+s[b:]
        s=s.replace('uint4* sm,int p,int hp,int count){','uint4* sm,int p,int hp,int count,const unsigned* mask){',1)
        s=s.replace('    const int* list=ids+', '    __shared__ unsigned mask[16];\n    if(threadIdx.x<16)mask[threadIdx.x]=0;\n    const int* list=ids+',1)
        s=s.replace('        if(step<0)continue;', '''        if(step<0)continue;
        if(step==0 && threadIdx.x<count){int id=list[threadIdx.x];atomicOr(mask+id/32,1u<<(id%32));}''',1)
        s=s.replace('pool,p,p,count);','pool,p,p,count,mask);',1)
    f=root/(name+'.cu');f.write_text(s)
    subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-lineinfo','-shared','-Xcompiler=-fPIC',str(f),'-o','/tmp/jzk_'+name+'.so'],check=True)
