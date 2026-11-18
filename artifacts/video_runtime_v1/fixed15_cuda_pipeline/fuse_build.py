from pathlib import Path
root=Path(__file__).parent
ex=(root/'shared_overlap.cu').read_text()
ta=(root/'packed_tail.cu').read_text()
ta=ta[ta.index('struct A'):ta.index('extern "C"')]
ta=ta.replace('__global__ void tail(', '__device__ __forceinline__ void tail(').replace('float* MU,float* CODE,float* LSE,int p,int hp,int count)', 'bf* out,const bf* basis,const float* exact,const float* le,uint4* sm,int p,int hp,int count)')
ta=ta.replace('__shared__ uint4 slab[2][words];','using Slot=uint4[words];\n    Slot* slab=reinterpret_cast<Slot*>(sm);')
i=ta.index('    #pragma unroll\n    for(int r=0;r<2;r++){\n        int i=h*4608')
ta=ta[:i]+'''    #pragma unroll
    for(int r=0;r<2;r++){
        float safe=fmaxf(den[r],1e-30f);
        float lt=den[r]>0?mx[r]+log2f(safe)+4.f:-CUDART_INF_F;
        int local=(row+8*r)%64;
        float rho=1.f/(1.f+ex(le[local]-lt));
        float coef[8];
        #pragma unroll
        for(int m=0;m<8;m++)coef[m]=__shfl_sync(~0u,ac.v[2*r+(m%2)]/safe,l-z+m/2);
        #pragma unroll
        for(int n=0;n<16;n++)
            #pragma unroll
            for(int j=0;j<2;j++){
                int dim=n*8+2*z+j;
                float a=0;
                #pragma unroll
                for(int m=0;m<8;m++)a=fmaf(coef[m],__bfloat162float(basis[(h*128+dim)*8+m]),a);
                float tv=am[n].v[2*r+j]/safe+a;
                float e=exact[local*128+dim];
                out[((row+8*r)*12+h)*128+dim]=__float2bfloat16(e*(1.f-rho)+tv*rho);
            }
    }
}
'''
i=ex.index('__global__ void exact')
ex=ex[:i]+'namespace tailops {\n'+ta+'\n}\n'+ex[i:]
ex='#include <math_constants.h>\n'+ex
ex=ex.replace('float* O,float* L,float* M,int count)', 'bf* O,const bf* summaries,const bf* basis,int* bad,int p,int count)')
i=ex.index('    #pragma unroll\n    for(int r=0;r<2;r++) {\n        // Emulate')
ex=ex[:i]+'''    __syncthreads();
    float* E=reinterpret_cast<float*>(pool+2656);
    float* L=E+8192;
    int all_ok=1;
'''+ex[i:]
ex=ex.replace('        int idx=(row+8*r)*12+h;','        int idx=(row+8*r)%64;\n        all_ok &= (s>=1.f && s<1.21e24f);').replace('if(pair==0){M[idx]=s;L[idx]=anchor[r]+log2f(s);}','if(pair==0)L[idx]=anchor[r]+log2f(s);').replace(' O[idx*128+n*8+pair*2+j]=z;',' E[idx*128+n*8+pair*2+j]=z;')
i=ex.index('\n}\nextern "C"')
ex=ex[:i]+'''
    int valid=__syncthreads_and(all_ok);
    if(!threadIdx.x)bad[h*72+t]=!valid;
    if(valid)tailops::tail(Q,summaries,nullptr,nullptr,ids,O,basis,E,L,pool,p,p,count);
}
extern "C" int run(const bf*q,const bf*k,const bf*v,const int*ids,bf*out,const bf*summaries,const bf*basis,int*bad,int p,int count,cudaStream_t stream){
    constexpr int bytes=2656*16+(8192+64)*4;
    static const auto opt=cudaFuncSetAttribute(exact,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(opt!=cudaSuccess)return opt;
    exact<<<dim3(72,12),128,bytes,stream>>>(q,k,v,ids,out,summaries,basis,bad,p,count);
    return cudaGetLastError();
}
'''
(root/'fused.cu').write_text(ex)
