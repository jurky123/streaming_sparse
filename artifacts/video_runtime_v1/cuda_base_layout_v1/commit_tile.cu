#include <cuda_runtime.h>

// One CTA copies 16 tokens of one head. The shared transpose feeds aligned
// fragment stores after the native cache write.
__global__ void commit_tile(const uint4* kg,const uint4* vg,uint4* k,uint4* v,
                            uint4* pk,uint4* pv,int start,int sk0,int sk1,int sv0,int sv1) {
    __shared__ ushort2 ks[16][65],vs[16][65];
    int page=blockIdx.x, head=blockIdx.y, strip=blockIdx.z, t=threadIdx.x;
    int slot=start+page;
    if (slot>=360) slot-=336;
    for(int j=t;j<256;j+=128) {
        int row=j/16, col=j%16;
        int token=page*64+strip*16+row;
        uint4 a=kg[(head*sk0+token*sk1)/8+col];
        uint4 b=vg[(head*sv0+token*sv1)/8+col];
        int dst=((slot*64+strip*16+row)*12+head)*16+col;
        k[dst]=a;v[dst]=b;
        #pragma unroll
        for(int n=0;n<4;++n) {
            ks[row][4*col+n]=reinterpret_cast<ushort2*>(&a)[n];
            vs[row][4*col+n]=reinterpret_cast<ushort2*>(&b)[n];
        }
    }
    __syncthreads();
    for(int j=t;j<256;j+=128) {
        uint4 a,b;
        #pragma unroll
        for(int u=0;u<8;++u) {
            int lane=j%32, red=2*(lane%4)+8*(u/2%2)+u%2;
            int col=8*(u/4)+lane/4;
            int kd=j/32*16+red, vd=j/32*16+col;
            ushort2 ak=ks[col][kd/2],bv=vs[red][vd/2];
            reinterpret_cast<unsigned short*>(&a)[u]=(kd%2)?ak.y:ak.x;
            reinterpret_cast<unsigned short*>(&b)[u]=(vd%2)?bv.y:bv.x;
        }
        int base=(slot*12+head)*1024;
        pk[base+j/32*128+strip*32+j%32]=a;
        pv[base+strip*256+j]=b;
    }
}
extern "C" int launch(const void* kg,const void* vg,void* k,void* v,void* pk,void* pv,
                      int start,int sk0,int sk1,int sv0,int sv1,cudaStream_t stream) {
    commit_tile<<<dim3(72,12,4),128,0,stream>>>((const uint4*)kg,(const uint4*)vg,
        (uint4*)k,(uint4*)v,(uint4*)pk,(uint4*)pv,start,sk0,sk1,sv0,sv1);
    return cudaGetLastError();
}
