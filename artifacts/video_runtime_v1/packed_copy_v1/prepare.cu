#include <cuda_runtime.h>
using halfbits=unsigned short;
__device__ __forceinline__ void copy_frag(const uint4* k,const uint4* v,uint4* pk,uint4* pv,int first,int id) {
    __shared__ __align__(16) halfbits keys[16][136],values[16][136];
    int page=first+id/48, head=id%12, strip=id/12%4;
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

__global__ void prepare(const uint4* k,const uint4* v,uint4* pk,uint4* pv,
                        const uint4* reps,const uint4* mean,const halfbits* detail,
                        uint4* summary,int pages,int hp) {
    int id=blockIdx.x, t=threadIdx.x,tiles=(pages+15)/16;
    if(id<72*48){copy_frag(k,v,pk,pv,pages-72,id);return;}
    id-=72*48;
    int h=id/(tiles*5),tile=id/5%tiles,strip=id%5;
    uint4* dst=summary+(h*tiles+tile)*1328;
    __shared__ __align__(16) halfbits buf[16][136];
    #pragma unroll
    for(int j=t;j<256;j+=128){
        int row=j/16,col=j%16,page=tile*16+row;
        uint4 val={};
        if(strip==4){if(page<pages)val=mean[(h*hp+page)*16+col];}
        else{
            int token=tile*64+strip*16+row;
            if(token<pages*4)val=reps[(h*hp*4+token)*16+col];
        }
        *reinterpret_cast<uint4*>(&buf[row][col*8])=val;
    }
    __syncthreads();
    #pragma unroll
    for(int j=t;j<256;j+=128){
        int lane=t%32, col=j/32*16,quad=lane/8;
        uint4 raw;
        if(strip==4){
            unsigned addr=__cvta_generic_to_shared(&buf[(quad%2)*8+lane%8][col+quad/2*8]);
            asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(raw.x),"=r"(raw.y),"=r"(raw.z),"=r"(raw.w):"r"(addr));
            dst[1024+j]=raw;
        }else{
            unsigned addr=__cvta_generic_to_shared(&buf[(quad/2)*8+lane%8][col+quad%2*8]);
            asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(raw.x),"=r"(raw.y),"=r"(raw.z),"=r"(raw.w):"r"(addr));
            dst[j/32*128+strip*32+lane]=raw;
        }
    }
    if(strip==4 && t<96){
        uint2 raw;
        int lane=t%32,u=t/32;
        #pragma unroll
        for(int b=0;b<4;b++){
            int page=tile*16+2*(lane%4)+8*(b/2)+b%2;
            reinterpret_cast<halfbits*>(&raw)[b]=page<pages?detail[((h*hp+page)*3+u)*8+lane/4]:0;
        }
        reinterpret_cast<uint2*>(dst+1280)[t]=raw;
    }
}
extern "C" int run(const uint4* k,const uint4* v,uint4* pk,uint4* pv,
                   const uint4* reps,const uint4* mean,const halfbits* detail,uint4* summary,
                   int pages,int hp,cudaStream_t stream){
    prepare<<<72*48+12*((pages+15)/16)*5,128,0,stream>>>(k,v,pk,pv,reps,mean,detail,summary,pages,hp);
    return cudaGetLastError();
}
