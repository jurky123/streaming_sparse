"""Two independent Q64 selections per CTA; share only identical prefix and summary loads."""
from pathlib import Path
import subprocess
root=Path(__file__).parent
s=(Path(__file__).parent/'control.cu').read_text()
a=s.index('__device__ __forceinline__ void tail(');b=s.index('extern "C" int run(',a)
head,body,end=s[:a],s[a:b],s[b:]
body=body.replace('    int h=blockIdx.y,t=blockIdx.x,l=threadIdx.x%32,w=threadIdx.x/32,','    int group=threadIdx.x/128, tid=threadIdx.x%128;\n    int h=blockIdx.y,t=blockIdx.x*2+group,l=tid%32,w=tid/32,')
body=body.replace('__shared__ unsigned mask[16];','__shared__ unsigned masks[2][16];\n    unsigned* mask=masks[group];').replace('if(threadIdx.x<16)mask[threadIdx.x]=0','if(tid<16)mask[tid]=0').replace('if(threadIdx.x<count){int id=ids[(h*72+t)*count+threadIdx.x]','if(tid<count){int id=ids[(h*72+t)*count+tid]')
body=body.replace('for(int i=threadIdx.x;i<words;i+=128)', 'for(int i=threadIdx.x;i<words;i+=256)')
body=body.replace('extern __shared__ uint4 pool[];', '''extern __shared__ uint4 common_pool[];
    int group=threadIdx.x/128,tid=threadIdx.x%128;
    uint4* pool=common_pool+group*(2656+2064);
    __shared__ int invalid[2];
    if(tid==0)invalid[group]=0;''')
body=body.replace('int t=blockIdx.x,h=blockIdx.y,w=threadIdx.x/32,l=threadIdx.x%32','int t=blockIdx.x*2+group,h=blockIdx.y,w=tid/32,l=tid%32')
body=body.replace('int slot=(step+1)&1;','''int slot=(step+1)&1;
            bool same=step>=0 && step<24;
            Slot* future=same?reinterpret_cast<Slot*>(common_pool):slots;''')
body=body.replace('for(int m=threadIdx.x;m<1024;m+=128)', 'for(int m=(same?threadIdx.x:tid);m<1024;m+=(same?256:128))')
body=body.replace('(&slots[slot]', '(&future[slot]')
body=body.replace('Slot& kv=slots[step&1];','Slot& kv=(step>=1 && step<=24?reinterpret_cast<Slot*>(common_pool):slots)[step&1];')
a=body.index('    int valid=__syncthreads_and')
body=body[:a]+'''    if(!all_ok)atomicOr(invalid+group,1);
    __syncthreads();
    if(tid==0)bad[h*72+t]=invalid[group];
    tailops::tail(Q,summaries,nullptr,nullptr,ids,O,basis,E,L,common_pool,p,p,count);
}
'''
end=end.replace('int bytes=2656*16+(8192+64)*4;', 'int bytes=2*(2656*16+(8192+64)*4);').replace('<<<dim3(72,12),128,bytes','<<<dim3(36,12),256,bytes',1)
f=root/'paired_cuda.cu';f.write_text(head+body+end)
subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_80','-O3','--fmad=false','-lineinfo','-Xptxas=-v','-shared','-Xcompiler=-fPIC',str(f),'-o','/tmp/jzk_paired_cuda.so'],check=True)
