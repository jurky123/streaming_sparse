#include <math_constants.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

using bf = __nv_bfloat16;
struct Four {float x[4];};
struct A {unsigned x[4];};
struct B {unsigned x[2];};

__device__ __forceinline__ void mma(Four &c, A a, B b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c.x[0]),"+f"(c.x[1]),"+f"(c.x[2]),"+f"(c.x[3])
        : "r"(a.x[0]),"r"(a.x[1]),"r"(a.x[2]),"r"(a.x[3]),"r"(b.x[0]),"r"(b.x[1]));
}
__device__ __forceinline__ unsigned packed(const bf *p) {
    return *reinterpret_cast<const unsigned*>(p);
}
__device__ __forceinline__ unsigned pairbf(float a,float b) {
    unsigned u;
    asm("cvt.rn.bf16x2.f32 %0,%2,%1;":"=r"(u):"f"(a),"f"(b));
    return u;
}
__device__ __forceinline__ float exp2ptx(float x) {
    float r; asm("ex2.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x));return r;
}
__device__ __forceinline__ int tile(int row,int col) {
    return row*128 + (col ^ ((row&7)*8));
}
__device__ __forceinline__ void stage(bf* dst,const bf* src,int page,int h) {
    #pragma unroll
    for(int i=threadIdx.x;i<64*16;i+=128) {
        int row=i/16, col=i%16*8;
        unsigned ptr=__cvta_generic_to_shared(dst+tile(row,col));
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(ptr),
            "l"(src+((page*64+row)*12+h)*128+col));
    }
}
namespace tailops {
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
__device__ __forceinline__ float divide(float a,float b){
    float c;asm("div.full.f32 %0,%1,%2;":"=f"(c):"f"(a),"f"(b));return c;
}
__device__ __forceinline__ unsigned tf(float x){
    unsigned u;asm("cvt.rna.tf32.f32 %0,%1;":"=r"(u):"f"(x));return u;
}
__device__ __forceinline__ void project(C& c,A a,B b){
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c.v[0]),"+f"(c.v[1]),"+f"(c.v[2]),"+f"(c.v[3])
        :"r"(a.v[0]),"r"(a.v[1]),"r"(a.v[2]),"r"(a.v[3]),"r"(b.v[0]),"r"(b.v[1]));
}
__device__ __forceinline__ void tail(const bf* Q,const bf* KEY,const bf* MEAN,const bf* DETAIL,const int* ids,
                     bf* out,const bf* basis,const float* exact,const float* le,uint4* sm,int p,int hp,int count){
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
    // One slab holds a 64-representative K tile, 16 means and three 16x8 detail tiles.
    constexpr int words=1024+256+48;
    using Slot=uint4[words];
    Slot* slab=reinterpret_cast<Slot*>(sm);
    int tiles=(p+15)/16;
    for(int step=0;step<tiles;step++){
        if(step){
            asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
        }
        if(step+1<tiles){
            int next=step+1;
            const uint4* input=reinterpret_cast<const uint4*>(KEY)+(h*tiles+next)*words;
            for(int i=threadIdx.x;i<words;i+=128){
                unsigned addr=__cvta_generic_to_shared(slab[next%2]+i);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(input+i));
            }
        }
        asm volatile("cp.async.commit_group;":::"memory");
        if(!step)continue;
        int pg=16*step;
        uint4* data=slab[step%2];
        C score[8]={};
        #pragma unroll
        for(int k=0;k<8;k++){
            #pragma unroll
            for(int n=0;n<8;n+=2){
                uint4 v=data[(k*4+n/2)*32+l];
                mma(score[n],q[k],B{{v.x,v.y}});mma(score[n+1],q[k],B{{v.z,v.w}});
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
                score[n].v[2*r]=a;score[n].v[2*r+1]=b;partial+=a;partial+=b;
            }
            partial+=__shfl_xor_sync(~0u,partial,2);partial+=__shfl_xor_sync(~0u,partial,1);
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
                float left[4]={(sums[0]+a[1])+b[1],(sums[0]-a[1])-b[1],(diffs[0]+a[1])-b[1],(diffs[0]-a[1])+b[1]};
                float right[4]={(sums[2]+a[3])+b[3],(sums[2]-a[3])-b[3],(diffs[2]+a[3])-b[3],(diffs[2]-a[3])+b[3]};
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
            uint4 raw=data[1024+n/2*32+l];
            B b=(n&1)?B{{raw.z,raw.w}}:B{{raw.x,raw.y}};
            #pragma unroll
            for(int j=0;j<4;j++)am[n].v[j]*=fade[j/2];
            mma(am[n],hi[0],b);mma(am[n],lo[0],b);
        }
        #pragma unroll
        for(int j=0;j<4;j++)ac.v[j]*=fade[j/2];
        #pragma unroll
        for(int u=0;u<3;u++){
            uint2 raw=reinterpret_cast<uint2*>(data+1280)[u*32+l];
            B b={{raw.x,raw.y}};
            C c={};mma(c,hi[1+u],b);mma(c,lo[1+u],b);
            #pragma unroll
            for(int j=0;j<4;j++)ac.v[j]+=c.v[j];
        }
    }
    A high,low;
    float inverse[2],rho[2];
    #pragma unroll
    for(int r=0;r<2;r++){
        float safe=fmaxf(den[r],1e-30f);
        inverse[r]=safe;
        float lt=den[r]>0?mx[r]+log2f(safe)+4.f:-CUDART_INF_F;
        int local=(row+8*r)%64;
        rho[r]=divide(1.f,1.f+ex(le[local]-lt));
        #pragma unroll
        for(int k=0;k<2;k++){
            float v=divide(ac.v[2*r+k],safe);
            high.v[2*k+r]=tf(v);
            low.v[2*k+r]=__float_as_uint(v-__uint_as_float(high.v[2*k+r]));
        }
    }
    #pragma unroll
    for(int n=0;n<16;n++){
        B b={{tf(__bfloat162float(basis[(h*128+n*8+g)*8+2*z])),
              tf(__bfloat162float(basis[(h*128+n*8+g)*8+2*z+1]))}};
        C proj={};project(proj,low,b);project(proj,high,b);
        #pragma unroll
        for(int r=0;r<2;r++)
            #pragma unroll
            for(int j=0;j<2;j++){
                int dim=n*8+2*z+j, local=(row+8*r)%64;
                float tv=divide(am[n].v[2*r+j],inverse[r])+proj.v[2*r+j];
                float e=exact[local*128+dim];
                out[((row+8*r)*12+h)*128+dim]=__float2bfloat16(e*(1.f-rho[r])+tv*rho[r]);
            }
    }

}

}
__global__ void exact(const bf* Q,const bf* K,const bf* V,const int* ids,
                      bf* O,const bf* summaries,const bf* basis,int* bad,int p,int count) {
    extern __shared__ uint4 pool[];
    using Slot=uint4[2][1024];
    Slot* slots=reinterpret_cast<Slot*>(pool);
    int t=blockIdx.x,h=blockIdx.y,w=threadIdx.x/32,l=threadIdx.x%32,g=l/4,pair=l%4;
    int row=t*64+w*16+g;
    const int* list=ids+(h*72+t)*count;
    A q[8];
    #pragma unroll
    for(int k=0;k<8;k++)
        #pragma unroll
        for(int i=0;i<4;i++)
            q[k].x[i]=packed(Q+((row+8*(i%2))*12+h)*128+16*k+(i/2)*8+2*pair);
    Four out[16]={},cols[8]={};float anchor[2]={};
    for(int step=-1;step<count;step++){
        int pg=step;
        if(step>=0) {
            asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
        }
        if(step+1<count){
            int page=list[step+1];
            int slot=(step+1)&1;
            const uint4* fk=reinterpret_cast<const uint4*>(K)+(page*12+h)*1024;
            const uint4* fv=reinterpret_cast<const uint4*>(V)+(page*12+h)*1024;
            for(int m=threadIdx.x;m<1024;m+=128){
                unsigned ak=__cvta_generic_to_shared(&slots[slot][0][m]);
                unsigned av=__cvta_generic_to_shared(&slots[slot][1][m]);
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ak),"l"(fk+m));
                asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(av),"l"(fv+m));
            }
        }
        asm volatile("cp.async.commit_group;":::"memory");
        if(step<0)continue;
        Slot& kv=slots[step&1];
        Four score[8]={};
        #pragma unroll
        for(int k=0;k<8;k++)
            #pragma unroll
            for(int n=0;n<8;n+=2) {
                uint4 u=reinterpret_cast<const uint4*>(kv[0])[(k*4+n/2)*32+l];
                B b[2]={{{u.x,u.y}},{{u.z,u.w}}};
                mma(score[n],q[k],b[0]);mma(score[n+1],q[k],b[1]);
            }
        #pragma unroll
        for(int r=0;r<2;r++) {
            float m=-INFINITY;
            #pragma unroll
            for(int n=0;n<8;n++)
                #pragma unroll
                for(int j=2*r;j<2*r+2;j++) {
                    score[n].x[j]=__fmul_rn(score[n].x[j],float(1.4426950408889634 / 11.313708498984761));
                    m=fmaxf(m,score[n].x[j]);
                }
            if(pg==0) {
                m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));
                anchor[r]=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));
            }
        }
        A high[4],low[4];
        #pragma unroll
        for(int n=0;n<8;n++)
            #pragma unroll
            for(int r=0;r<2;r++) {
                float x=exp2ptx(score[n].x[r*2]-anchor[r]);
                float y=exp2ptx(score[n].x[r*2+1]-anchor[r]);
                cols[n].x[r*2]+=x; cols[n].x[r*2+1]+=y;
                high[n/2].x[2*(n%2)+r]=pairbf(x,y);
                low[n/2].x[2*(n%2)+r]=pairbf(x-__bfloat162float(__float2bfloat16(x)),y-__bfloat162float(__float2bfloat16(y)));
            }
        #pragma unroll
        for(int n=0;n<16;n+=2) {
            Four tmp[2]={}; B b[4][2];
            #pragma unroll
            for(int k=0;k<4;k++) {
                uint4 u=reinterpret_cast<const uint4*>(kv[1])[(k*8+n/2)*32+l];
                b[k][0]=B{{u.x,u.y}}; b[k][1]=B{{u.z,u.w}};
                mma(tmp[0],high[k],b[k][0]); mma(tmp[1],high[k],b[k][1]);
            }
            #pragma unroll
            for(int k=0;k<4;k++) {
                mma(tmp[0],low[k],b[k][0]); mma(tmp[1],low[k],b[k][1]);
            }
            #pragma unroll
            for(int t=0;t<2;t++)
                #pragma unroll
                for(int j=0;j<4;j++)out[n+t].x[j]+=tmp[t].x[j];
        }
    }
    __syncthreads();
    float* E=reinterpret_cast<float*>(pool+2656);
    float* L=E+8192;
    int all_ok=1;
    #pragma unroll
    for(int r=0;r<2;r++) {
        // Emulate the four column warps of the reference reduction.
        float partial[4];
        #pragma unroll
        for(int n=0;n<4;n++) {
            float v=cols[n].x[2*r]+cols[n].x[2*r+1];
            v+=cols[n+4].x[2*r];
            v+=cols[n+4].x[2*r+1];
            v+=__shfl_xor_sync(0xffffffff,v,2);
            partial[n]=v+__shfl_xor_sync(0xffffffff,v,1);
        }
        float s=(partial[0]+partial[2])+(partial[1]+partial[3]);
        int idx=(row+8*r)%64;
        all_ok &= (s>=1.f && s<1.21e24f);
        if(pair==0)L[idx]=anchor[r]+log2f(s);
        #pragma unroll
        for(int n=0;n<16;n++)
            #pragma unroll
            for(int j=0;j<2;j++){float z; asm("div.full.f32 %0,%1,%2;":"=f"(z):"f"(out[n].x[2*r+j]),"f"(s)); E[idx*128+n*8+pair*2+j]=z;}
    }
    int valid=__syncthreads_and(all_ok);
    if(!threadIdx.x)bad[h*72+t]=!valid;
    if(valid)tailops::tail(Q,summaries,nullptr,nullptr,ids,O,basis,E,L,pool,p,p,count);
}
extern "C" int run(const bf*q,const bf*k,const bf*v,const int*ids,bf*out,const bf*summaries,const bf*basis,int*bad,int p,int count,cudaStream_t stream){
    constexpr int bytes=2656*16+(8192+64)*4;
    static thread_local int configured=-1;
    int device;
    auto status=cudaGetDevice(&device);
    if(status!=cudaSuccess)return status;
    if(configured!=device){
        status=cudaFuncSetAttribute(exact,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
        if(status!=cudaSuccess)return status;
        configured=device;
    }
    exact<<<dim3(72,12),128,bytes,stream>>>(q,k,v,ids,out,summaries,basis,bad,p,count);
    return cudaGetLastError();
}



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
extern "C" int commit_fixed15(const void* kg,const void* vg,void* k,void* v,void* pk,void* pv,
                      int start,int sk0,int sk1,int sv0,int sv1,cudaStream_t stream) {
    commit_tile<<<dim3(72,12,4),128,0,stream>>>((const uint4*)kg,(const uint4*)vg,
        (uint4*)k,(uint4*)v,(uint4*)pk,(uint4*)pv,start,sk0,sk1,sv0,sv1);
    return cudaGetLastError();
}
