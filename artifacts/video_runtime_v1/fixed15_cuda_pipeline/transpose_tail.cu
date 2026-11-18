#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
using bf=__nv_bfloat16;
struct A {unsigned v[4];};
struct B {unsigned v[2];};
struct C {float v[4];};
__device__ __forceinline__ void mma(C &c,A a,B b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c.v[0]),"+f"(c.v[1]),"+f"(c.v[2]),"+f"(c.v[3])
        :"r"(a.v[0]),"r"(a.v[1]),"r"(a.v[2]),"r"(a.v[3]),"r"(b.v[0]),"r"(b.v[1]));
}
__device__ __forceinline__ unsigned pair(float x,float y){
    unsigned v;asm("cvt.rn.bf16x2.f32 %0,%2,%1;":"=r"(v):"f"(x),"f"(y));return v;
}
__device__ __forceinline__ B operand(const bf* v,int stride,int pg,int end,int pair_lane){
    B b;
    #pragma unroll
    for(int j=0;j<2;j++){
        int n=pg+2*pair_lane+8*j;
        float x=n<end?__bfloat162float(v[n*stride]):0.f;
        float y=n+1<end?__bfloat162float(v[(n+1)*stride]):0.f;
        b.v[j]=pair(x,y);
    }
    return b;
}
__device__ __forceinline__ float ex(float v){float o;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(o):"f"(v));return o;}
__global__ void tail(const bf* Q,const bf* KEY,const bf* MEAN,const bf* DETAIL,const int* ids,
                     float* MU,float* CODE,float* LSE,int p,int hp,int count){
    int h=blockIdx.y,t=blockIdx.x,l=threadIdx.x%32,w=threadIdx.x/32,g=l/4,z=l%4,row=t*64+w*16+g;
    __shared__ unsigned mask[16];
    if(threadIdx.x<16)mask[threadIdx.x]=0;
    __syncthreads();
    if(threadIdx.x<count){int id=ids[(h*72+t)*count+threadIdx.x];atomicOr(mask+id/32,1u<<(id%32));}
    __syncthreads();
    A q[8];C am[16]={},ac={};float mx[2]={-1e30f,-1e30f},den[2]={};
    #pragma unroll
    for(int k=0;k<8;k++)
        #pragma unroll
        for(int j=0;j<4;j++)
            q[k].v[j]=*reinterpret_cast<const unsigned*>(Q+((row+8*(j%2))*12+h)*128+k*16+j/2*8+z*2);
    for(int pg=16;pg<p;pg+=16){ // Fixed15 pages 0..15 are always exact.
        C score[8]={};
        #pragma unroll
        for(int n=0;n<8;n++){
            int rep=pg*4+n*8+g;
            #pragma unroll
            for(int k=0;k<8;k++){
                B b={};
                if(rep<p*4){
                    const bf* ptr=KEY+((h*hp*4+rep)*128)+16*k+2*z;
                    b.v[0]=*reinterpret_cast<const unsigned*>(ptr);
                    b.v[1]=*reinterpret_cast<const unsigned*>(ptr+8);
                }
                mma(score[n],q[k],b);
            }
        }
        unsigned bits=mask[pg/32];
        A hi[4],lo[4];float fade[2];
        #pragma unroll
        for(int r=0;r<2;r++){
            float nm=mx[r];
            #pragma unroll
            for(int n=0;n<8;n++)
                #pragma unroll
                for(int j=0;j<2;j++){
                    int page=pg+n*2+z/2;
                    float v=(page<p && !(bits&(1u<<(page%32))))?__fmul_rn(score[n].v[r*2+j],0.12751743082459868f):-CUDART_INF_F;
                    score[n].v[r*2+j]=v;nm=fmaxf(nm,v);
                }
            nm=fmaxf(nm,__shfl_xor_sync(~0u,nm,1));nm=fmaxf(nm,__shfl_xor_sync(~0u,nm,2));
            fade[r]=ex(mx[r]-nm);mx[r]=nm;
            float partial=0;
            #pragma unroll
            for(int n=0;n<8;n++){
                float a=ex(score[n].v[2*r]-nm),b=ex(score[n].v[2*r+1]-nm);
                score[n].v[2*r]=a;score[n].v[2*r+1]=b;partial+=a+b;
            }
            partial+=__shfl_xor_sync(~0u,partial,1);partial+=__shfl_xor_sync(~0u,partial,2);
            den[r]=den[r]*fade[r]+partial;
            #pragma unroll
            for(int chunk=0;chunk<2;chunk++){
                // Each lane of a four-thread row needs the two pages of N fragment (4*chunk+z).
                float a[4],b[4];
                #pragma unroll
                for(int j=0;j<4;j++){
                    a[j]=score[4*chunk+j].v[2*r];b[j]=score[4*chunk+j].v[2*r+1];
                }
                #pragma unroll
                for(int bit=1;bit<=2;bit*=2){
                    #pragma unroll
                    for(int j=0;j<4;j++)if(!(j&bit)){
                        float x=__shfl_xor_sync(~0u,(z&bit)?a[j]:a[j+bit],bit);
                        float y=__shfl_xor_sync(~0u,(z&bit)?b[j]:b[j+bit],bit);
                        if(z&bit){a[j]=x;b[j]=y;}else{a[j+bit]=x;b[j+bit]=y;}
                    }
                }
                float sums[4],diffs[4];
                #pragma unroll
                for(int j=0;j<4;j++){sums[j]=a[j]+b[j];diffs[j]=a[j]-b[j];}
                float left[4]={sums[0]+sums[1],sums[0]-sums[1],diffs[0]+diffs[1],diffs[0]-diffs[1]};
                float right[4]={sums[2]+sums[3],sums[2]-sums[3],diffs[2]+diffs[3],diffs[2]-diffs[3]};
                #pragma unroll
                for(int u=0;u<4;u++){
                    float a=left[u],b=right[u];
                    hi[u].v[2*chunk+r]=pair(a,b);
                    lo[u].v[2*chunk+r]=pair(a-__bfloat162float(__float2bfloat16(a)),b-__bfloat162float(__float2bfloat16(b)));
                }
            }
        }
        #pragma unroll
        for(int n=0;n<16;n++){
            B b=operand(MEAN+(h*hp)*128+n*8+g,128,pg,p,z);
            C c={};mma(c,hi[0],b);mma(c,lo[0],b);
            #pragma unroll
            for(int j=0;j<4;j++)am[n].v[j]=am[n].v[j]*fade[j/2]+c.v[j];
        }
        #pragma unroll
        for(int j=0;j<4;j++)ac.v[j]*=fade[j/2];
        #pragma unroll
        for(int u=0;u<3;u++){
            B b=operand(DETAIL+(h*hp*3+u)*8+g,24,pg,p,z);
            C c={};mma(c,hi[1+u],b);mma(c,lo[1+u],b);
            #pragma unroll
            for(int j=0;j<4;j++)ac.v[j]+=c.v[j];
        }
    }
    #pragma unroll
    for(int r=0;r<2;r++){
        int i=h*4608+row+8*r;
        float safe=fmaxf(den[r],1e-30f);
        if(!z)LSE[i]=den[r]>0?mx[r]+log2f(safe)+4:-CUDART_INF_F;
        #pragma unroll
        for(int j=0;j<2;j++){
            CODE[i*8+2*z+j]=ac.v[r*2+j]/safe;
            #pragma unroll
            for(int n=0;n<16;n++)MU[i*128+n*8+2*z+j]=am[n].v[r*2+j]/safe;
        }
    }
}
extern "C" int run(const bf*q,const bf*key,const bf*mean,const bf*detail,const int*ids,
                   float*mu,float*code,float*lse,int p,int hp,int count,cudaStream_t s){
    tail<<<dim3(72,12),128,0,s>>>(q,key,mean,detail,ids,mu,code,lse,p,hp,count);
    return cudaGetLastError();
}
