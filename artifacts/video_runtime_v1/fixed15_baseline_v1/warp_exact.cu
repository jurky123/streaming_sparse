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
                      float* O,float* L,float* M,int count) {
    __shared__ bf kv[2][64*128];
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
    stage(kv[0],K,list[0],h);
    asm volatile("cp.async.commit_group;":::"memory");
    for(int pg=0;pg<count;pg++) {
        stage(kv[1],V,list[pg],h);
        asm volatile("cp.async.commit_group; cp.async.wait_group 1;":::"memory");
        __syncthreads();
        Four score[8]={};
        #pragma unroll
        for(int k=0;k<8;k++)
            #pragma unroll
            for(int n=0;n<8;n+=2) {
                unsigned addr=__cvta_generic_to_shared(kv[0]+tile((n+l/16)*8+l%8,k*16+(l/8%2)*8));
                B b[2];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(b[0].x[0]),"=r"(b[0].x[1]),"=r"(b[1].x[0]),"=r"(b[1].x[1]):"r"(addr));
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
        asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
        if(pg+1<count) stage(kv[0],K,list[pg+1],h);
        asm volatile("cp.async.commit_group;":::"memory");
        #pragma unroll
        for(int n=0;n<16;n+=2) {
            Four tmp[2]={}; B b[4][2];
            #pragma unroll
            for(int k=0;k<4;k++) {
                unsigned addr=__cvta_generic_to_shared(kv[1]+tile(k*16+l%16,(n+l/16)*8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
                    :"=r"(b[k][0].x[0]),"=r"(b[k][0].x[1]),"=r"(b[k][1].x[0]),"=r"(b[k][1].x[1]):"r"(addr));
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
        __syncthreads();
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
extern "C" int run(const bf*q,const bf*k,const bf*v,const int*ids,float*out,float*lse,float*mass,int count,cudaStream_t s){
    exact<<<dim3(72,12),128,0,s>>>(q,k,v,ids,out,lse,mass,count);
    return cudaGetLastError();
}
