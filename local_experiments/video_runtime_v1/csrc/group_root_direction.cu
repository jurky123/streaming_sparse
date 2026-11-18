#include <cuda_runtime.h>

// One CTA owns one of the original 32 interleaved transpose-GEMV chains.
// Group directions synchronize through a small 32-by-32 buffer between iterations.
template<int N,bool FIRST,bool FINAL>
__global__ void chain(const float* xc,const long* axis,const float* dir,float* scratch,float* scores) {
    constexpr int R=N/32;
    int group=blockIdx.y,chain=blockIdx.x,tid=threadIdx.x;
    __shared__ float x[R][33],projection[R],v[32];
    for(int i=tid;i<R*32;i+=128)
        x[i/32][i%32]=xc[(group*N+(i/32)*32+chain)*32+i%32];
    if(!FIRST && tid<32) v[tid]=dir[group*32+tid];
    __syncthreads();
    for(int row=tid;row<R;row+=128) {
        float p;
        if constexpr(FIRST) p=x[row][axis[group]];
        else {
            float a=0.f,b=0.f;
            #pragma unroll
            for(int j=0;j<32;j+=2) {
                a=__fmaf_rn(x[row][j],v[j],a); b=__fmaf_rn(x[row][j+1],v[j+1],b);
            }
            p=__fadd_rn(a,b);
        }
        if constexpr(FINAL) scores[group*N+row*32+chain]=p;
        else projection[row]=p;
    }
    if constexpr(!FINAL) {
        __syncthreads();
        if(tid<32) {
            float a=0.f;
            #pragma unroll 4
            for(int row=0;row<R;row++) a=__fmaf_rn(x[row][tid],projection[row],a);
            scratch[(group*32+chain)*32+tid]=a;
        }
    }
}
__global__ void finish(const float* scratch,float* dir) {
    __shared__ float p[32][32];
    int tid=threadIdx.x,lane=tid%32;
    for(int i=tid;i<1024;i+=128) p[i/32][i%32]=scratch[blockIdx.x*1024+i];
    __syncthreads();
    for(int d=16;d;d/=2) {
        for(int i=tid;i<32*d;i+=128) p[i/32][i%32]=__fadd_rn(p[i/32][i%32],p[i/32+d][i%32]);
        __syncthreads();
    }
    if(tid<32) {
        float v=p[0][lane],s=__fmul_rn(v,v);
        for(int d=1;d<32;d*=2) s=__fadd_rn(s,__shfl_down_sync(0xffffffff,s,d));
        dir[blockIdx.x*32+lane]=__fdiv_rn(v,fmaxf(__fsqrt_rn(__shfl_sync(0xffffffff,s,0)),1.e-8f));
    }
}
template<int N>
void run(const float* x,const long* axis,float* scratch,float* dir,float* out,int g,cudaStream_t stream) {
    dim3 grid(32,g);
    chain<N,true,false><<<grid,128,0,stream>>>(x,axis,dir,scratch,out);
    finish<<<g,128,0,stream>>>(scratch,dir);
    for(int i=1;i<4;i++) {
        chain<N,false,false><<<grid,128,0,stream>>>(x,axis,dir,scratch,out);
        finish<<<g,128,0,stream>>>(scratch,dir);
    }
    chain<N,false,true><<<grid,128,0,stream>>>(x,axis,dir,scratch,out);
}
extern "C" void root_direction(const float* x,const long* axis,float* scratch,float* dir,float* out,int g,int n,cudaStream_t stream) {
    if(n==1536)run<1536>(x,axis,scratch,dir,out,g,stream);
    if(n==4608)run<4608>(x,axis,scratch,dir,out,g,stream);
}
