#include <math_constants.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

using bf = __nv_bfloat16;
struct Four {float x[4];};
struct A {unsigned x[4];};
struct B {unsigned x[2];};

__device__ __forceinline__ void mma(Four &c, A a, B b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c.x[0]),"+f"(c.x[1]),"+f"(c.x[2]),"+f"(c.x[3])
        : "r"(a.x[0]),"r"(a.x[1]),"r"(a.x[2]),"r"(a.x[3]),"r"(b.x[0]),"r"(b.x[1]));
}
__device__ __forceinline__ unsigned packed(const bf *p) {
    return *reinterpret_cast<const unsigned*>(p);
}
__device__ __forceinline__ unsigned pairbf(float a,float b) {
    unsigned u;
    asm("cvt.rn.bf16x2.f32 %0,%2,%1;":"=r"(u):"f"(a),"f"(b));
    return u;
}
__device__ __forceinline__ float exp2ptx(float x) {
    float r; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x));return r;
}
__global__ void exact(const bf* Q,const bf* K,const bf* V,bf* O,int count) {
    extern __shared__ uint4 pool[];
    using Slot=uint4[2][1024];
    Slot* slots=reinterpret_cast<Slot*>(pool);
    int t=blockIdx.x,h=blockIdx.y,w=threadIdx.x/32,l=threadIdx.x%32,g=l/4,pair=l%4;
    int row=t*64+w*16+g;

    A q[8];
    #pragma unroll
    for(int k=0;k<8;k++)
        #pragma unroll
        for(int i=0;i<4;i++)
            q[k].x[i]=packed(Q+((row+8*(i%2))*12+h)*128+16*k+(i/2)*8+2*pair);
    Four out[16]={};float mx[2]={-INFINITY,-INFINITY},den[2]={};
    for(int step=-1;step<count;step++){
        int pg=step;
        if(step>=0) {
            asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
        }
        if(step+1<count){
            int page=step+1;
            int slot=(step+1)&1;
            const uint4* fk=reinterpret_cast<const uint4*>(K)+(page*12+h)*1024;
            const uint4* fv=reinterpret_cast<const uint4*>(V)+(page*12+h)*1024;
            for(int m=threadIdx.x;m<1024;m+=128){
                unsigned ak=__cvta_generic_to_shared(&slots[slot][0][m]);
                unsigned av=__cvta_generic_to_shared(&slots[slot][1][m]);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ak),"l"(fk+m));
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(av),"l"(fv+m));
            }
        }
        asm volatile("cp.async.commit_group;":::"memory");
        if(step<0)continue;
        Slot& kv=slots[step&1];
        Four score[8]={};
        #pragma unroll
        for(int k=0;k<8;k++)
            #pragma unroll
            for(int n=0;n<8;n+=2) {
                uint4 u=reinterpret_cast<const uint4*>(kv[0])[(k*4+n/2)*32+l];
                B b[2]={{{u.x,u.y}},{{u.z,u.w}}};
                mma(score[n],q[k],b[0]);mma(score[n+1],q[k],b[1]);
            }

        A high[4];
        #pragma unroll
        for(int r=0;r<2;r++) {
            float nm=mx[r];
            #pragma unroll
            for(int n=0;n<8;n++) {
                #pragma unroll
                for(int j=0;j<2;j++) {
                    float &z=score[n].x[2*r+j];
                    z*=float(1.4426950408889634 / 11.313708498984761);
                    nm=fmaxf(nm,z);
                }
            }
            nm=fmaxf(nm,__shfl_xor_sync(0xffffffff,nm,2));
            nm=fmaxf(nm,__shfl_xor_sync(0xffffffff,nm,1));
            float fade=exp2ptx(mx[r]-nm);mx[r]=nm;
            float sum=0;
            #pragma unroll
            for(int n=0;n<8;n++){
                float x=exp2ptx(score[n].x[2*r]-nm),y=exp2ptx(score[n].x[2*r+1]-nm);
                high[n/2].x[2*(n%2)+r]=pairbf(x,y);
                sum+=x+y;
            }
            den[r]=den[r]*fade+sum;
            #pragma unroll
            for(int n=0;n<16;n++){
                out[n].x[2*r]*=fade;out[n].x[2*r+1]*=fade;
            }
        }
        #pragma unroll
        for(int k=0;k<4;k++)
            #pragma unroll
            for(int n=0;n<16;n+=2) {
                uint4 u=kv[1][(k*8+n/2)*32+l];
                mma(out[n],high[k],B{{u.x,u.y}});mma(out[n+1],high[k],B{{u.z,u.w}});
            }
    }
    #pragma unroll
    for(int r=0;r<2;r++) {
        float s=den[r];s+=__shfl_xor_sync(0xffffffff,s,2);s+=__shfl_xor_sync(0xffffffff,s,1);
        #pragma unroll
        for(int n=0;n<16;n++)
            #pragma unroll
            for(int j=0;j<2;j++)O[((row+8*r)*12+h)*128+n*8+2*pair+j]=__float2bfloat16(out[n].x[2*r+j]/s);
    }
}
extern "C" int run(const bf*q,const bf*k,const bf*v,bf*out,int p,cudaStream_t stream){
    constexpr int bytes=65536;
    static const auto opt=cudaFuncSetAttribute(exact,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(opt!=cudaSuccess)return opt;
    exact<<<dim3(72,12),128,bytes,stream>>>(q,k,v,out,p);
    return cudaGetLastError();
}
