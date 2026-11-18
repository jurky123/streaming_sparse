#include <cuda_runtime.h>
using halfbits=unsigned short;
__global__ void copy_frag(const uint4* k,const uint4* v,uint4* pk,uint4* pv,int first) {
    __shared__ __align__(16) halfbits keys[16][136],values[16][136];
    int page=first+blockIdx.z, head=blockIdx.x, strip=blockIdx.y;
    int t=threadIdx.x, lane=t%32;
    #pragma unroll
    for(int i=t;i<256;i+=128) {
        int row=i/16,col=i%16;
        int src=(((page*64+strip*16+row)*12+head)*16+col);
        *reinterpret_cast<uint4*>(&keys[row][col*8])=k[src];
        *reinterpret_cast<uint4*>(&values[row][col*8])=v[src];
    }
    __syncthreads();
    #pragma unroll
    for(int j=t;j<256;j+=128){
        int col=(j/32)*16,quad=lane/8;
        unsigned ka=__cvta_generic_to_shared(&keys[(quad/2)*8+lane%8][col+(quad%2)*8]);
        unsigned va=__cvta_generic_to_shared(&values[(quad%2)*8+lane%8][col+(quad/2)*8]);
        uint4 a,b;
        asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"r"(ka));
        asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(b.x),"=r"(b.y),"=r"(b.z),"=r"(b.w):"r"(va));
        int base=(page*12+head)*1024;
        pk[base+(j/32)*128+strip*32+lane]=a;
        pv[base+strip*256+j]=b;
    }
}
extern "C" int run(const uint4* k,const uint4* v,uint4* pk,uint4* pv,int start,cudaStream_t stream) {
    copy_frag<<<dim3(12,4,72),128,0,stream>>>(k,v,pk,pv,start);
    return cudaGetLastError();
}
