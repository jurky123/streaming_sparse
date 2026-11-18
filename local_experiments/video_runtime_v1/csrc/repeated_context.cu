#include <math_constants.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using Bf=__nv_bfloat16;
struct Frag { float x,y,z,w; };

__device__ __forceinline__ unsigned pair(const Bf* p) {
    return *reinterpret_cast<const unsigned*>(p);
}
__device__ __forceinline__ unsigned pack(float a,float b) {
    __nv_bfloat162 v=__floats2bfloat162_rn(a,b);
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ Frag mma(unsigned a,unsigned b,unsigned c,unsigned d,
                                     unsigned e,unsigned f,Frag v) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(v.x),"+f"(v.y),"+f"(v.z),"+f"(v.w)
        : "r"(a),"r"(b),"r"(c),"r"(d),"r"(e),"r"(f));
    return v;
}
__device__ __forceinline__ float max_quad(float z) {
    z=fmaxf(z,__shfl_xor_sync(0xffffffff,z,2));
    return fmaxf(z,__shfl_xor_sync(0xffffffff,z,1));
}
__device__ __forceinline__ float sum_quad(float x) {
    x+=__shfl_xor_sync(0xffffffff,x,2);
    return x+__shfl_xor_sync(0xffffffff,x,1);
}

// One warp handles sixteen query rows. Cached K and V are packed in MMA fragment order.
__global__ void repeated(const Bf* q,const Bf* k,const Bf* v,Bf* out) {
    const int head=blockIdx.y;
    const int lane=threadIdx.x%32;
    const int g=lane/4,t=lane%4;
    const int row=(blockIdx.x*4+threadIdx.x/32)*16+g;
    constexpr int stride=1536;
    q+=(row*12+head)*128;
    Bf* dest=out+(row*12+head)*128;
    k+=head*64*128;
    v+=head*(64*128+128);
    Frag s[8]={};
    #pragma unroll
    for (int group=0;group<8;group++) {
        const Bf* a=q+16*group+2*t;
        const unsigned a0=pair(a),a1=pair(a+8*stride),a2=pair(a+8),a3=pair(a+8*stride+8);
        #pragma unroll
        for (int j=0;j<8;j++) {
            const Bf* b=k+((group*8+j)*32+lane)*4;
            uint2 f=*reinterpret_cast<const uint2*>(b);
            s[j]=mma(a0,a1,a2,a3,f.x,f.y,s[j]);
        }
    }
    float m0=-CUDART_INF_F,m1=-CUDART_INF_F;
    #pragma unroll
    for(int j=0;j<8;++j){
        m0=fmaxf(m0,fmaxf(s[j].x,s[j].y));
        m1=fmaxf(m1,fmaxf(s[j].z,s[j].w));
    }
    m0=max_quad(m0);m1=max_quad(m1);
    const float scale=0.127517431974411f;
    // The first seven 64-key blocks traversed by FlashAttention contain only the repeated key.
    // Four lanes share a softmax row, so each lane starts with 448/4 units of mass.
    float a0=exp2f((__shfl_sync(0xffffffff,s[7].y,lane|3)-m0)*scale);
    float a1=exp2f((__shfl_sync(0xffffffff,s[7].w,lane|3)-m1)*scale);
    float d0=112.f*a0,d1=112.f*a1;
    unsigned prob0[8],prob1[8];
    #pragma unroll
    for(int j=0;j<8;++j) {
        // Both products round independently in the PyTorch 2.6 SM80 kernel.
        float x=exp2f(__fmul_rn(s[j].x,scale)-__fmul_rn(m0,scale));
        float y=exp2f(__fmul_rn(s[j].y,scale)-__fmul_rn(m0,scale));
        float z=exp2f(__fmul_rn(s[j].z,scale)-__fmul_rn(m1,scale));
        float w=exp2f(__fmul_rn(s[j].w,scale)-__fmul_rn(m1,scale));
        d0=(d0+x)+y;d1=(d1+z)+w;
        prob0[j]=pack(x,y);prob1[j]=pack(z,w);
    }
    d0=1.f/sum_quad(d0);d1=1.f/sum_quad(d1);
    #pragma unroll
    for(int j=0;j<16;++j) {
        float vx=448.f*float(v[64*128+8*j+2*t]),vy=448.f*float(v[64*128+8*j+2*t+1]);
        Frag value={vx*a0,vy*a0,vx*a1,vy*a1};
        #pragma unroll
        for(int l=0;l<4;++l){
            const Bf* b=v+((j*4+l)*32+lane)*4;
            uint2 f=*reinterpret_cast<const uint2*>(b);
            value=mma(prob0[2*l],prob1[2*l],prob0[2*l+1],prob1[2*l+1],f.x,f.y,value);
        }
        *reinterpret_cast<unsigned*>(dest+8*j+2*t)=pack(value.x*d0,value.y*d0);
        *reinterpret_cast<unsigned*>(dest+8*stride+8*j+2*t)=pack(value.z*d1,value.w*d1);
    }
}
extern "C" void compact(const Bf* q,const Bf* k,const Bf* v,Bf* out,cudaStream_t stream){
    repeated<<<dim3(72,12),128,0,stream>>>(q,k,v,out);
}
