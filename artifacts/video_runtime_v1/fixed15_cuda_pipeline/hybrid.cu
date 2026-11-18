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
__device__ __forceinline__ int tile(int row,int col) {
    return row*128 + (col ^ ((row&7)*8));
}
__device__ __forceinline__ void stage(bf* dst,const bf* src,int page,int h) {
    #pragma unroll
    for(int i=threadIdx.x;i<64*16;i+=128) {
        int row=i/16, col=i%16*8;
        unsigned ptr=__cvta_generic_to_shared(dst+tile(row,col));
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(ptr),
            "l"(src+((page*64+row)*12+h)*128+col));
    }
}
__global__ void exact(const bf* Q,const bf* K,const bf* V,const int* ids,
                      float* O,float* L,float* M,int count,const bf* rawk,const bf* rawv,int hist) {
    extern __shared__ uint4 pool[];
    using Slot=uint4[2][1024];
    Slot* slots=reinterpret_cast<Slot*>(pool);
    int t=blockIdx.x,h=blockIdx.y,w=threadIdx.x/32,l=threadIdx.x%32,g=l/4,pair=l%4;
    int row=t*64+w*16+g;
    const int* list=ids+(h*72+t)*count;
    A q[8];
    #pragma unroll
    for(int k=0;k<8;k++)
        #pragma unroll
        for(int i=0;i<4;i++)
            q[k].x[i]=packed(Q+((row+8*(i%2))*12+h)*128+16*k+(i/2)*8+2*pair);
    Four out[16]={},cols[8]={};float anchor[2]={};
    for(int step=-1;step<count;step++){
        int pg=step;
        if(step>=0) {
            asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
        }
        if(step+1<count){
            int page=list[step+1];
            int slot=(step+1)&1;
            const uint4* fk=reinterpret_cast<const uint4*>(K)+(page*12+h)*1024;
            const uint4* fv=reinterpret_cast<const uint4*>(V)+(page*12+h)*1024;
            if(page>=hist){
                stage(reinterpret_cast<bf*>(slots[slot][0]),rawk,page,h);
                stage(reinterpret_cast<bf*>(slots[slot][1]),rawv,page,h);
            }else{
            for(int m=threadIdx.x;m<1024;m+=128){
                unsigned ak=__cvta_generic_to_shared(&slots[slot][0][m]);
                unsigned av=__cvta_generic_to_shared(&slots[slot][1][m]);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ak),"l"(fk+m));
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(av),"l"(fv+m));
            }
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
                B b[2];
                if(list[pg]>=hist){
                unsigned addr=__cvta_generic_to_shared(reinterpret_cast<bf*>(kv[0])+tile((n+l/16)*8+l%8,k*16+(l/8%2)*8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(b[0].x[0]),"=r"(b[0].x[1]),"=r"(b[1].x[0]),"=r"(b[1].x[1]):"r"(addr));
                }else{
                    uint4 u=kv[0][(k*4+n/2)*32+l];
                    b[0]=B{{u.x,u.y}};b[1]=B{{u.z,u.w}};
                }
                mma(score[n],q[k],b[0]);mma(score[n+1],q[k],b[1]);
            }
        #pragma unroll
        for(int r=0;r<2;r++) {
            float m=-INFINITY;
            #pragma unroll
            for(int n=0;n<8;n++)
                #pragma unroll
                for(int j=2*r;j<2*r+2;j++) {
                    score[n].x[j]=__fmul_rn(score[n].x[j],float(1.4426950408889634 / 11.313708498984761));
                    m=fmaxf(m,score[n].x[j]);
                }
            if(pg==0) {
                m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));
                anchor[r]=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));
            }
        }
        A high[4],low[4];
        #pragma unroll
        for(int n=0;n<8;n++)
            #pragma unroll
            for(int r=0;r<2;r++) {
                float x=exp2ptx(score[n].x[r*2]-anchor[r]);
                float y=exp2ptx(score[n].x[r*2+1]-anchor[r]);
                cols[n].x[r*2]+=x; cols[n].x[r*2+1]+=y;
                high[n/2].x[2*(n%2)+r]=pairbf(x,y);
                low[n/2].x[2*(n%2)+r]=pairbf(x-__bfloat162float(__float2bfloat16(x)),y-__bfloat162float(__float2bfloat16(y)));
            }
        #pragma unroll
        for(int n=0;n<16;n+=2) {
            Four tmp[2]={}; B b[4][2];
            #pragma unroll
            for(int k=0;k<4;k++) {
                if(list[pg]>=hist){
                unsigned addr=__cvta_generic_to_shared(reinterpret_cast<bf*>(kv[1])+tile(k*16+l%16,(n+l/16)*8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(b[k][0].x[0]),"=r"(b[k][0].x[1]),"=r"(b[k][1].x[0]),"=r"(b[k][1].x[1]):"r"(addr));
                }else{
                    uint4 u=kv[1][(k*8+n/2)*32+l];
                    b[k][0]=B{{u.x,u.y}};b[k][1]=B{{u.z,u.w}};
                }
                mma(tmp[0],high[k],b[k][0]); mma(tmp[1],high[k],b[k][1]);
            }
            #pragma unroll
            for(int k=0;k<4;k++) {
                mma(tmp[0],low[k],b[k][0]); mma(tmp[1],low[k],b[k][1]);
            }
            #pragma unroll
            for(int t=0;t<2;t++)
                #pragma unroll
                for(int j=0;j<4;j++)out[n+t].x[j]+=tmp[t].x[j];
        }
    }
    #pragma unroll
    for(int r=0;r<2;r++) {
        // Emulate the four column warps of the reference reduction.
        float partial[4];
        #pragma unroll
        for(int n=0;n<4;n++) {
            float v=cols[n].x[2*r]+cols[n].x[2*r+1];
            v+=cols[n+4].x[2*r];
            v+=cols[n+4].x[2*r+1];
            v+=__shfl_xor_sync(0xffffffff,v,2);
            partial[n]=v+__shfl_xor_sync(0xffffffff,v,1);
        }
        float s=(partial[0]+partial[2])+(partial[1]+partial[3]);
        int idx=(row+8*r)*12+h;
        if(pair==0){M[idx]=s;L[idx]=anchor[r]+log2f(s);}
        #pragma unroll
        for(int n=0;n<16;n++)
            #pragma unroll
            for(int j=0;j<2;j++){float z; asm("div.full.f32 %0,%1,%2;":"=f"(z):"f"(out[n].x[2*r+j]),"f"(s)); O[idx*128+n*8+pair*2+j]=z;}
    }
}
extern "C" int run(const bf*q,const bf*k,const bf*v,const int*ids,float*out,float*lse,float*mass,int count,cudaStream_t s,const bf* rawk,const bf* rawv,int hist){
    static const auto optin=cudaFuncSetAttribute(exact,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
    if(optin!=cudaSuccess)return optin;
    exact<<<dim3(72,12),128,65536,s>>>(q,k,v,ids,out,lse,mass,count,rawk,rawv,hist);
    return cudaGetLastError();
}
