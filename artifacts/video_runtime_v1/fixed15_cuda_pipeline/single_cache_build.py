from pathlib import Path
from inspect import getsource
from local_experiments.review_v1.tuned_attention import _tail
from local_experiments.gap_exact_round_kernel import _attn_fwd
p=Path(__file__).parent
s=(p/'paired_k.cu').read_text()
s=s.replace('+(page*12+h)*1024;', '+page*12*1024+h*16;').replace('("l"(fk+m))','("l"(fk+m))').replace('"l"(fk+m)', '"l"(fk+(m/16)*192+(m%16))').replace('"l"(fv+m)', '"l"(fv+(m/16)*192+(m%16))')
s+='''
template<bool DECODE> __global__ void permute(const bf* k,const bf* v,bf* dk,bf* dv,int start,int history){
    int page=start+blockIdx.x,h=blockIdx.y;
    __shared__ bf kbuf[8192],vbuf[8192];
    int base=(page*64*12+h)*128;
    for(int m=threadIdx.x;m<1024;m+=blockDim.x){
        int src=base+(m/16)*1536+(m%16)*8;
        reinterpret_cast<uint4*>(kbuf)[m]=*reinterpret_cast<const uint4*>(k+src);
        reinterpret_cast<uint4*>(vbuf)[m]=*reinterpret_cast<const uint4*>(v+src);
    }
    __syncthreads();
    for(int i=threadIdx.x;i<8192;i+=blockDim.x){
        int l=i/8%32,red=2*(l%4)+8*(i/2%2)+i%2,col=8*(i/4%2)+l/4;
        int kr=16*(i/1024)+red,kc=16*(i/256%4)+col;
        int vr=16*(i/2048)+red,vc=16*(i/256%8)+col;
        int dst=base+i/128*1536+i%128;
        if(DECODE){
            if(page<history){
                dk[base+kc*1536+kr]=kbuf[i];dv[base+vr*1536+vc]=vbuf[i];
            }else{
                dk[dst]=kbuf[i];dv[dst]=vbuf[i];
            }
        }else{
            dk[dst]=kbuf[kc*128+kr];dv[dst]=vbuf[vr*128+vc];
        }
    }
}
extern "C" int layout(bf*k,bf*v,bf*dk,bf*dv,int start,int pages,int history,int decode,cudaStream_t s){
    if(decode)permute<true><<<dim3(pages,12),256,0,s>>>(k,v,dk,dv,start,history);
    else permute<false><<<dim3(pages,12),256,0,s>>>(k,v,dk,dv,start,history);
    return cudaGetLastError();
}
'''
(p/'single_cache.cu').write_text(s)
s=getsource(_attn_fwd.fn).replace('def _attn_fwd(', 'def packed_online(')
i=s.index('        K_ptrs =');j=s.index('        k = tl.load',i)
s=s[:i]+'''        kindex=frag_index(offs_d[:,None],offs_n[None,:],4)
        vindex=frag_index(offs_n[:,None],offs_d[None,:],8)
        base=(idx_n*64*H+idx_h)*D
        K_ptrs=K+base+kindex//128*HD+kindex%128
        V_ptrs=V+base+vindex//128*HD+vindex%128
'''+s[j:]
a=getsource(_tail.fn).replace('def _tail(', 'def packed_tail(').replace('_attn_fwd(', 'packed_online(')
s='''import math
import triton
import triton as tr
import triton.language as tl
from local_experiments.review_v1.tuned_attention import _av,_anchored_exact
@tr.jit
def frag_index(red,col,groups:tl.constexpr):
 lane=4*(col%8)+(red%8)//2
 return ((((red//16*groups+col//16)*32+lane)*2+col%16//8)*4+red%16//8*2+red%2)
'''+s+'\n'+a
f=(p/'repair.py').read_text()
s+=f[f.index('@tr.jit'):].replace('_tail(', 'packed_tail(').replace('ANCHOR_SCALE=True','ANCHOR_SCALE=False')
(p/'single_retry.py').write_text(s)
