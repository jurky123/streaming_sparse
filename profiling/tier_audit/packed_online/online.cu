#include "../../../local_experiments/video_runtime_v1/csrc/fixed15_attention.cu"

__global__ void online(const bf* Q,const bf* K,const bf* V,const int* ids,
                       float* E,float* L,int count) {
    extern __shared__ uint4 shared[];
    using Slot=uint4[2][1024];
    Slot* slots=reinterpret_cast<Slot*>(shared);
    int t=blockIdx.x,h=blockIdx.y,w=threadIdx.x/32,l=threadIdx.x%32,g=l/4,pair=l%4;
    int row=t*64+w*16+g;
    const int* list=ids+(h*72+t)*count;
    A q[8];
    #pragma unroll
    for(int k=0;k<8;k++)
        #pragma unroll
        for(int i=0;i<4;i++)
            q[k].x[i]=packed(Q+((row+8*(i%2))*12+h)*128+16*k+(i/2)*8+2*pair);
    Four out[16]={};
    float peak[2]={-INFINITY,-INFINITY}, mass[2]={};
    copy_page(slots[0],K,V,(list[0]*12+h)*1024);
    asm volatile("cp.async.commit_group;":::"memory");
    for(int step=0;step<count;step++) {
        asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
        if(step+1<count)copy_page(slots[(step+1)%2],K,V,(list[step+1]*12+h)*1024);
        asm volatile("cp.async.commit_group;":::"memory");
        Slot& kv=slots[step%2];
        Four score[8]={};
        #pragma unroll
        for(int k=0;k<8;k++)
            #pragma unroll
            for(int n=0;n<8;n+=2){
                uint4 u=kv[0][(k*4+n/2)*32+l];
                mma(score[n],q[k],B{{u.x,u.y}});
                mma(score[n+1],q[k],B{{u.z,u.w}});
            }
        A high[4],low[4];
        float fade[2];
        #pragma unroll
        for(int r=0;r<2;r++){
            float mx=peak[r];
            #pragma unroll
            for(int n=0;n<8;n++)
                #pragma unroll
                for(int j=0;j<2;j++){
                    float& v=score[n].x[2*r+j];
                    v=__fmul_rn(v,float(1.4426950408889634 / 11.313708498984761));
                    mx=fmaxf(mx,v);
                }
            mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
            mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));
            fade[r]=exp2ptx(peak[r]-mx);
            peak[r]=mx;
            float partial[4];
            #pragma unroll
            for(int n=0;n<8;n++){
                float x=exp2ptx(score[n].x[2*r]-mx), y=exp2ptx(score[n].x[2*r+1]-mx);
                if(n<4)partial[n]=x+y;
                else partial[n-4]=(partial[n-4]+x)+y;
                high[n/2].x[2*(n%2)+r]=pairbf(x,y);
                low[n/2].x[2*(n%2)+r]=pairbf(x-__bfloat162float(__float2bfloat16(x)),y-__bfloat162float(__float2bfloat16(y)));
            }
            #pragma unroll
            for(int n=0;n<4;n++){
                partial[n]+=__shfl_xor_sync(0xffffffff,partial[n],2);
                partial[n]+=__shfl_xor_sync(0xffffffff,partial[n],1);
            }
            mass[r]=__fmaf_rn(mass[r],fade[r],(partial[0]+partial[2])+(partial[1]+partial[3]));
        }
        #pragma unroll
        for(int n=0;n<16;n+=2){
            Four hi[2]={};
            B b[4][2];
            #pragma unroll
            for(int k=0;k<4;k++){
                uint4 u=kv[1][(k*8+n/2)*32+l];
                b[k][0]=B{{u.x,u.y}}; b[k][1]=B{{u.z,u.w}};
                #pragma unroll
                for(int j=0;j<2;j++)mma(hi[j],high[k],b[k][j]);
            }
            #pragma unroll
            for(int k=0;k<4;k++)
                #pragma unroll
                for(int j=0;j<2;j++)mma(hi[j],low[k],b[k][j]);
            #pragma unroll
            for(int j=0;j<2;j++)
                #pragma unroll
                for(int i=0;i<4;i++)
                    out[n+j].x[i]=__fmaf_rn(out[n+j].x[i],fade[i/2],hi[j].x[i]);
        }
    }
    #pragma unroll
    for(int r=0;r<2;r++){
        int token=row+r*8;
        if(pair==0)L[h*4608+token]=peak[r]+log2f(mass[r]);
        #pragma unroll
        for(int n=0;n<16;n++)
            #pragma unroll
            for(int j=0;j<2;j++){
                float v;
                asm("div.full.f32 %0,%1,%2;":"=f"(v):"f"(out[n].x[2*r+j]),"f"(mass[r]));
                E[(token*12+h)*128+n*8+pair*2+j]=v;
            }
    }
}
extern "C" int launch_online(const bf*q,const bf*k,const bf*v,const int*ids,float*e,float*l,int count,cudaStream_t stream){
    cudaFuncSetAttribute(online,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
    online<<<dim3(72,12),128,65536,stream>>>(q,k,v,ids,e,l,count);
    return cudaGetLastError();
}
