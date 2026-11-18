#include <cuda_bf16.h>
#include <cuda_runtime.h>
union Vec { uint4 u; __nv_bfloat16 b[8]; };
__global__ void generate(const uint4* x,const uint4* w,uint4* native,const uint4* v,
                         uint4* pk,uint4* pv,const float* mean,const double2* freq,int gh,int gw,
                         int start,float eps) {
    __shared__ __nv_bfloat16 key[16][136],val[16][136];
    int page=blockIdx.x,head=blockIdx.y,strip=blockIdx.z,t=threadIdx.x;
    for(int j=t;j<256;j+=128) {
        int row=j/16, col=j%16;
        int token=page*64+strip*16+row;
        int idx=(token*12+head)*16+col;
        Vec a,weight,b;a.u=x[idx];weight.u=w[head*16+col];b.u=v[idx];
        float inv=rsqrtf(mean[token]+eps);
        #pragma unroll
        for(int i=0;i<4;++i) {
            int pair=col*4+i;
            int pos=pair<22?token/(gh*gw)+start:pair<43?(token/gw)%gh:token%gw;
            double2 f=freq[pos*64+pair];
            double re=__bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(__bfloat162float(a.b[2*i])*inv))*__bfloat162float(weight.b[2*i])));
            double im=__bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(__bfloat162float(a.b[2*i+1])*inv))*__bfloat162float(weight.b[2*i+1])));
            a.b[2*i]=__float2bfloat16_rn(__double2float_rn(re*f.x-im*f.y));
            a.b[2*i+1]=__float2bfloat16_rn(__double2float_rn(re*f.y+im*f.x));
        }
        native[idx]=a.u;
        #pragma unroll
        for(int u=0;u<8;++u){key[row][col*8+u]=a.b[u];val[row][col*8+u]=b.b[u];}
    }
    __syncthreads();
    for(int j=t;j<256;j+=128) {
        Vec a,b;
        #pragma unroll
        for(int u=0;u<8;++u) {
            int lane=j%32, red=2*(lane%4)+8*(u/2%2)+u%2, col=8*(u/4)+lane/4;
            a.b[u]=key[col][j/32*16+red];b.b[u]=val[red][j/32*16+col];
        }
        int base=(page*12+head)*1024;
        pk[base+j/32*128+strip*32+j%32]=a.u;pv[base+strip*256+j]=b.u;
    }
}
extern "C" int emit_tile(const uint4* x,const uint4* w,uint4* native,const uint4* v,uint4* pk,
                         uint4* pv,const float* mean,const double2* f,int gh,int gw,int start,float eps,cudaStream_t s) {
    generate<<<dim3(72,12,4),128,0,s>>>(x,w,native,v,pk,pv,mean,f,gh,gw,start,eps);
    return cudaGetLastError();
}
