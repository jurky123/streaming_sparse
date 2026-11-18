#include <cuda_runtime.h>

// Two FP32 FMA accumulators reproduce the deployed cuBLAS small-node GEMVs.
// Shared padding keeps parallel reads of consecutive rows on distinct banks.
template<int N,bool SQUARE>
__device__ void column_mean(float (*x)[33],float (&tmp)[4][32]) {
    constexpr int Y=N<64?1:4;
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    if (warp<Y) {
        float part[4]={};
        #pragma unroll
        for(int i=0;i<N/(4*Y);i++) {
            #pragma unroll
            for(int j=0;j<4;j++) {
                float v=x[(4*i+j)*Y+warp][lane];
                part[j]=__fadd_rn(part[j],SQUARE?__fmul_rn(v,v):v);
            }
        }
        tmp[warp][lane]=__fadd_rn(__fadd_rn(__fadd_rn(part[0],part[1]),part[2]),part[3]);
    }
    __syncthreads();
    for(int d=Y/2;d;d/=2) {
        if(warp<d) tmp[warp][lane]=__fadd_rn(tmp[warp][lane],tmp[warp+d][lane]);
        __syncthreads();
    }
    if(!warp) tmp[0][lane]=__fmul_rn(tmp[0][lane],1.f/N);
    __syncthreads();
}

template<int N,bool RAW=false>
__global__ void direction_kernel(const float* input, const long* axes, float* score) {
    __shared__ float small_x[N<512?N:1][33], projection[N], direction[32], partial[4][32];
    extern __shared__ float large_x[][33];
    float (*x)[33]=N<512?small_x:large_x;
    __shared__ int max_axis;
    int group=blockIdx.x, tid=threadIdx.x, lane=tid%32, warp=tid/32;
    for(int i=tid;i<N*32;i+=blockDim.x)
        x[i/32][i%32]=input[group*N*32+i];
    __syncthreads();
    int axis;
    if constexpr(RAW) {
        column_mean<N,false>(x,partial);
        for(int i=tid;i<N*32;i+=blockDim.x)
            x[i/32][i%32]=__fsub_rn(x[i/32][i%32],partial[0][i%32]);
        __syncthreads();
        column_mean<N,true>(x,partial);
        if(!warp) {
            float v=partial[0][lane];
            for(int d=16;d;d/=2) v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));
            unsigned mask=__ballot_sync(0xffffffff,v==partial[0][lane]);
            if(!lane) max_axis=__ffs(mask)-1;
        }
        __syncthreads();
        axis=max_axis;
    } else axis=axes[group];
    for(int t=tid;t<N;t+=blockDim.x) projection[t]=x[t][axis];
    __syncthreads();
    #pragma unroll
    for(int iter=0;iter<4;iter++) {
        constexpr int chains=N==512?4:2;
        if(warp<chains) {
            float s=0.f;
            #pragma unroll
            for(int j=0;j<N/chains;j++) {
                int t=N==32?warp*16+j:N==256?(j/64)*128+warp*64+j%64:chains*j+warp;
                s=__fmaf_rn(x[t][lane],projection[t],s);
            }
            partial[warp][lane]=s;
        }
        __syncthreads();
        if(warp==0) {
            float v;
            if constexpr(N==512) v=__fadd_rn(__fadd_rn(partial[0][lane],partial[2][lane]),__fadd_rn(partial[1][lane],partial[3][lane]));
            else v=__fadd_rn(partial[0][lane],partial[1][lane]);
            float s=__fmul_rn(v,v);
            #pragma unroll
            for(int d=1;d<32;d*=2) s=__fadd_rn(s,__shfl_down_sync(0xffffffff,s,d));
            float norm=fmaxf(__fsqrt_rn(__shfl_sync(0xffffffff,s,0)),1.e-8f);
            direction[lane]=__fdiv_rn(v,norm);
        }
        __syncthreads();
        for(int t=tid;t<N;t+=blockDim.x) {
            float a=0.f,b=0.f;
            #pragma unroll
            for(int j=0;j<32;j+=2) {
                a=__fmaf_rn(x[t][j],direction[j],a);
                b=__fmaf_rn(x[t][j+1],direction[j+1],b);
            }
            projection[t]=__fadd_rn(a,b);
        }
        __syncthreads();
    }
    for(int t=tid;t<N;t+=blockDim.x) score[group*N+t]=projection[t];
}
extern "C" void exact_direction(const float* x,const long* axes,float* out,int groups,int n,cudaStream_t stream) {
    if(n==32) direction_kernel<32><<<groups,64,0,stream>>>(x,axes,out);
    if(n==64) direction_kernel<64><<<groups,64,0,stream>>>(x,axes,out);
    if(n==128) direction_kernel<128><<<groups,128,0,stream>>>(x,axes,out);
}

extern "C" void raw_direction(const float* x,float* out,int groups,int n,cudaStream_t stream) {
    if(n==32) direction_kernel<32,true><<<groups,64,0,stream>>>(x,nullptr,out);
    if(n==64) direction_kernel<64,true><<<groups,128,0,stream>>>(x,nullptr,out);
    if(n==256) direction_kernel<256,true><<<groups,256,0,stream>>>(x,nullptr,out);
    if(n==512) direction_kernel<512,true><<<groups,256,512*33*sizeof(float),stream>>>(x,nullptr,out);
    if(n==128) direction_kernel<128,true><<<groups,128,0,stream>>>(x,nullptr,out);
}

extern "C" int enable_large_direction() {
    return cudaFuncSetAttribute(direction_kernel<512,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,512*33*sizeof(float));
}
