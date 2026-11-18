// A CTA owns one 16-token representative. Change thread ownership between
// column moments and row scores, retaining the deployed FP32 reduction trees.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <initializer_list>
using BF = __nv_bfloat16;
constexpr unsigned FULL = 0xffffffff;

template<bool Feature>
__device__ __forceinline__ float sum_rows(const float (&x)[16]) {
    float a[8], b[4];
    if (Feature) {
        #pragma unroll
        for (int g=0;g<4;++g)
            b[g]=(x[4*g]+x[4*g+2])+(x[4*g+1]+x[4*g+3]);
        return (b[0]+b[2])+(b[1]+b[3]);
    }
    #pragma unroll
    for(int i=0;i<8;++i) a[i]=x[i]+x[i+8];
    #pragma unroll
    for(int i=0;i<4;++i) b[i]=a[2*i]+a[2*i+1];
    return (b[0]+b[2])+(b[1]+b[3]);
}

template<int Width>
__device__ __forceinline__ float warp_sum(float a) {
    #pragma unroll
    for(int i=Width/2;i;i/=2) a+=__shfl_xor_sync(FULL,a,i);
    return a;
}

__global__ void pool_rep(const BF* K,const BF* V,const float* F,const float* WK,
                         const float* WV,const float* WF,float* PK,float* PV,float* AUX,
                         int P,int kh,int kp,int kt,int vh,int vp,int vt) {
    int rep=blockIdx.x,page=rep/4,r=rep%4,h=page/P,p=page%P,col=threadIdx.x;
    __shared__ BF sk[16][128],sv[16][128];
    __shared__ float km[128],vm[128],var[128],z[16],vnorm[16],aa[16];
    float k[16],v[16];
    #pragma unroll
    for(int t=0;t<16;++t) {
        BF key=K[h*kh+p*kp+(16*r+t)*kt+col],value=V[h*vh+p*vp+(16*r+t)*vt+col];
        sk[t][col]=key;sv[t][col]=value;
        k[t]=__bfloat162float(key);v[t]=__bfloat162float(value);
    }
    float mk=sum_rows<false>(k)*.0625f,mv=sum_rows<false>(v)*.0625f;
    km[col]=mk;vm[col]=mv;
    AUX[rep*265+col]=mk;AUX[rep*265+128+col]=mv;
    float kd[16];
    #pragma unroll
    for(int i=0;i<16;++i){float delta=k[i]-mk;kd[i]=delta*delta;}
    var[col]=sum_rows<false>(kd)*.0625f;
    __syncthreads();

    int sub=col%16,first=8*sub;
    #pragma unroll
    for(int half=0;half<2;++half) {
        int t=col/16+8*half;
        float dk=0.f,dv=0.f,norm=0.f,feature=0.f;
        #pragma unroll
        for(int j=0;j<8;++j) {
            int d=first+j;
            float x=__bfloat162float(sk[t][d])-km[d],y=__bfloat162float(sv[t][d])-vm[d];
            float a=x*WK[(4*h+r)*128+d],b=y*WV[(4*h+r)*128+d],c=y*y;
            dk=j?dk+a:a;dv=j?dv+b:b;norm=j?norm+c:c;
        }
        #pragma unroll
        for(int j=0;j<4;++j) {
            int d=4*(sub%8)+j;
            float v=F[(16*rep+t)*32+d]*WF[(4*h+r)*32+d];
            feature=j?feature+v:v;
        }
        dk=warp_sum<16>(dk);dv=warp_sum<16>(dv);norm=warp_sum<16>(norm);
        feature=warp_sum<8>(feature);
        if(!sub){z[t]=(dk+dv)+feature;vnorm[t]=norm;}
    }
    float vv=warp_sum<16>(var[col])*.0625f;
    if(!sub) AUX[rep*265+256+col/16]=vv;
    __syncthreads();
    if(col<32) {
        int lane=col%16;
        float v=z[lane],center=v,norm=vnorm[lane];
        for(int i : {2,1,8,4}) center+=__shfl_xor_sync(FULL,center,i);
        for(int i : {8,1,4,2}) norm+=__shfl_xor_sync(FULL,norm,i);
        if(!col) AUX[rep*265+264]=__fsqrt_rn(norm*.0625f);
        v=2.f*tanhf((v-center*.0625f)*.5f);
        float mx=v;
        #pragma unroll
        for(int i=8;i;i/=2) mx=fmaxf(mx,__shfl_xor_sync(FULL,mx,i));
        float a;
        asm("ex2.approx.f32 %0,%1;" : "=f"(a) : "f"((v-mx)*1.4426950408889634f));
        float den=a;
        for(int i : {2,1,8,4}) den+=__shfl_xor_sync(FULL,den,i);
        float fraction;
        asm("div.full.f32 %0,%1,%2;" : "=f"(fraction) : "f"(a),"f"(den));
        if(col<16) aa[lane]=fraction;
    }
    __syncthreads();
    #pragma unroll
    for(int i=0;i<16;++i){k[i]*=aa[i];v[i]*=aa[i];}
    PK[rep*128+col]=sum_rows<true>(k);PV[rep*128+col]=sum_rows<true>(v);
}
extern "C" void run_pool(const BF* k,const BF* v,const float* f,const float* wk,const float* wv,
                         const float* wf,float* pk,float* pv,float* aux,int h,int p,
                         int kh,int kp,int kt,int vh,int vp,int vt,cudaStream_t stream) {
    pool_rep<<<h*p*4,128,0,stream>>>(k,v,f,wk,wv,wf,pk,pv,aux,p,kh,kp,kt,vh,vp,vt);
}
