#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sum4(const float (&x)[4]) {
    return ((x[0]+x[1])+x[2])+x[3];
}
__device__ __forceinline__ float exp_div(float x, float den) {
    float a, p;
    asm("ex2.approx.f32 %0,%1;" : "=f"(a) : "f"(x*1.4426950408889634f));
    asm("div.full.f32 %0,%1,%2;" : "=f"(p) : "f"(a),"f"(den));
    return p;
}

// Adjacent lanes read adjacent pages. Each lane owns the four representatives,
// while the eight warps share the resident statistics for two probes apiece.
__global__ void risk_pages(const float* q2, const float* qn, const float4* z,
                           const float2* norm, const __nv_bfloat16* st, const float* theta,
                           const float* c, float* out, int pages, int tiles) {
    int tile=blockIdx.y, h=tile/tiles, lane=threadIdx.x, warp=threadIdx.y;
    int page=32*blockIdx.x+lane;
    __shared__ float values[16][32];
    if (page<pages) {
        float stats[4][12];
        const __nv_bfloat16* ptr=st+(h*pages+page)*48;
        #pragma unroll
        for(int r=0;r<4;++r) {
            #pragma unroll
            for(int g=0;g<12;++g) stats[r][g]=__bfloat162float(ptr[r*12+g]);
        }
        #pragma unroll
        for(int j=0;j<2;++j) {
            int m=warp+8*j,row=tile*16+m;
            float2 nd=norm[row];
            float4 logits=z[row*pages+page];
            float prob[4]={exp_div(logits.x-nd.x,nd.y),exp_div(logits.y-nd.x,nd.y),
                           exp_div(logits.z-nd.x,nd.y),exp_div(logits.w-nd.x,nd.y)};
            float alpha=sum4(prob),a[4],b[4],e[4],f[4];
            #pragma unroll
            for(int r=0;r<4;++r) {
                float u=0.f;
                #pragma unroll
                for(int g=0;g<8;++g) u=fmaf(q2[row*8+g],stats[r][g],u);
                u=fmaxf(u,0.f);
                float beta=qn[row]*stats[r][8];
                a[r]=fabsf(prob[r]-alpha*.25f)*stats[r][9];
                b[r]=(prob[r]*sqrtf(u))*stats[r][10];
                e[r]=prob[r]*beta;
                f[r]=prob[r]*(u+beta*beta);
            }
            float w=c[h*pages+page];
            float f0=(alpha*stats[0][11]+sum4(a))*theta[h*4];
            float f1=sum4(b)*theta[h*4+1];
            float f2=(w*sum4(e))*theta[h*4+2];
            float f3=(w*sum4(f))*theta[h*4+3];
            values[m][lane]=((f0+f1)+f2)+f3;
        }
    }
    __syncthreads();
    if (!warp && page<pages) {
        float x[8];
        #pragma unroll
        for(int i=0;i<8;++i) x[i]=values[i][lane]+values[i+8][lane];
        #pragma unroll
        for(int i=0;i<4;++i) x[i]+=x[i+4];
        out[tile*pages+page]=((x[0]+x[2])+(x[1]+x[3]))*.0625f;
    }
}
extern "C" void run_pages(const float* q2,const float* qn,const float4* z,const float2* norm,
                          const __nv_bfloat16* st,const float* theta,const float* c,float* out,
                          int heads,int tiles,int pages,cudaStream_t stream) {
    risk_pages<<<dim3((pages+31)/32,heads*tiles),dim3(32,8),0,stream>>>(
        q2,qn,z,norm,st,theta,c,out,pages,tiles);
}
