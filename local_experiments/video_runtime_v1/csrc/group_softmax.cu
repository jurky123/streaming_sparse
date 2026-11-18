// FP32 spatial softmax with PyTorch 2.6's per-column accumulation order.
// Reference: pytorch commit 2236df1770800ffea5697b11b0bb0d910b2e59e1,
// aten/src/ATen/native/cuda/SoftMax.cu (BSD-3-Clause).
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

template<bool MAXIMUM>
__device__ float reduce_column(float value, float* shared, int lane, int column) {
    float* row = shared + column * 32;
    __syncthreads();
    row[lane] = value;
    for (int offset=16; offset>0; offset/=2) {
        __syncthreads();
        if (lane<offset) {
            float a=row[lane], b=row[lane+offset];
            row[lane] = MAXIMUM ? (a<b ? b : a) : a+b;
        }
    }
    __syncthreads();
    return row[0];
}

template<bool COALESCED>
__global__ void spatial32(float* output, const float* input, int length) {
    extern __shared__ float shared[];
    int lane=COALESCED ? threadIdx.y : threadIdx.x;
    int column=COALESCED ? threadIdx.x : threadIdx.y;
    int columns=COALESCED ? blockDim.x : blockDim.y;
    int col=blockIdx.y*columns+column;
    int base=blockIdx.x*length*32+col;
    float maximum=-FLT_MAX;
    #pragma unroll 1
    for (int d=lane; d<length; d+=32) {
        float x=input[base+d*32];maximum=maximum<x ? x : maximum;
    }
    maximum=reduce_column<true>(maximum,shared,lane,column);
    float sum=0;
    #pragma unroll 1
    for (int d=lane; d<length; d+=32) sum+=expf(input[base+d*32]-maximum);
    sum=reduce_column<false>(sum,shared,lane,column);
    #pragma unroll 1
    for (int d=lane; d<length; d+=32) output[base+d*32]=expf(input[base+d*32]-maximum)/sum;
}

extern "C" void group_softmax(float* output, const float* input, int outer, int length, int columns, int coalesced, cudaStream_t stream) {
    if (coalesced) spatial32<true><<<dim3(outer,32/columns),dim3(columns,32),32*columns*sizeof(float),stream>>>(output,input,length);
    else spatial32<false><<<dim3(outer,32/columns),dim3(32,columns),32*columns*sizeof(float),stream>>>(output,input,length);
}

// Stage each column once. Only the sum needs the original 32 sequential chains.
template<int C>
__global__ void staged_columns(float* output, const float* input) {
    constexpr int N=4608, W=256/C, LANES=32;
    extern __shared__ float values[];
    __shared__ float partial[256];
    int tid=threadIdx.x, col=tid%C, lane=tid/C;
    int base=blockIdx.x*N*32+blockIdx.y*C+col;
    float maximum=-FLT_MAX;
    for(int i=lane;i<N;i+=W) {
        float x=input[base+i*32];
        values[i*C+col]=x;
        maximum=fmaxf(maximum,x);
    }
    partial[tid]=maximum;
    __syncthreads();
    for(int off=W/2;off;off/=2) {
        if(lane<off) partial[tid]=fmaxf(partial[tid],partial[tid+off*C]);
        __syncthreads();
    }
    maximum=partial[col];
    for(int i=lane;i<N;i+=W)
        values[i*C+col]=expf(values[i*C+col]-maximum);
    __syncthreads();
    if(lane<LANES) {
        float sum=0;
        for(int i=lane;i<N;i+=LANES) sum+=values[i*C+col];
        partial[tid]=sum;
    }
    __syncthreads();
    for(int off=LANES/2;off;off/=2) {
        if(lane<off) partial[tid]+=partial[tid+off*C];
        __syncthreads();
    }
    float divisor=partial[col];
    for(int i=lane;i<N;i+=W) output[base+i*32]=values[i*C+col]/divisor;
}

extern "C" void group_staged(float* output,const float* input,int outer,int columns,cudaStream_t stream) {
    if(columns==1) staged_columns<1><<<dim3(outer,32),256,4608*sizeof(float),stream>>>(output,input);
    else if(columns==2) staged_columns<2><<<dim3(outer,16),256,9216*sizeof(float),stream>>>(output,input);
    else {
        cudaFuncSetAttribute(staged_columns<4>,cudaFuncAttributeMaxDynamicSharedMemorySize,73728);
        staged_columns<4><<<dim3(outer,8),256,73728,stream>>>(output,input);
    }
}
