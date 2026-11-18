// FP32 reduction order follows PyTorch's vectorized_layer_norm_kernel for D=1536.
// See PYTORCH_LICENSE.txt for the upstream license.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using BF=__nv_bfloat16;
struct alignas(8) Four { BF x[4]; };
struct Moments { float mean, m2; };

__device__ Moments combine(Moments left,Moments right,float count) {
    float diff=left.mean-right.mean;
    return {fmaf(.5f,right.mean,.5f*left.mean),
            fmaf(diff*diff*count,.5f,right.m2+left.m2)};
}

enum class Stage { Modulate, AddModulate, GatedAffine };
template<Stage stage>
__global__ void normalize_modulate(const Four* x,const Four* s,const Four* t,Four* out,
                                   float eps,int frame_len,int ss,int ts, const Four* residual, Four* sum,const Four* gate,int gs) {
    __shared__ Moments warps[4];
    int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    int row=blockIdx.x;
    Four input[3];
    Moments v{0.f,0.f};
    #pragma unroll
    for(int j=0;j<3;j++) {
        int index=row*384+j*128+tid;
        input[j]=x[index];
        if (stage!=Stage::Modulate) {
            Four addend=residual[index];
            if (stage==Stage::GatedAffine) {
                Four g=gate[(row/frame_len)*gs+j*128+tid];
                #pragma unroll
                for (int i=0;i<4;i++)
                    addend.x[i]=__float2bfloat16_rn(__bfloat162float(addend.x[i])*__bfloat162float(g.x[i]));
            }
            #pragma unroll
            for (int i=0;i<4;i++) {
                input[j].x[i]=__float2bfloat16_rn(__bfloat162float(input[j].x[i])+__bfloat162float(addend.x[i]));
            }
            sum[index]=input[j];
        }
        #pragma unroll
        for(int i=0;i<4;i++) {
            float f=__bfloat162float(input[j].x[i]);
            float delta=f-v.mean;
            v.mean=fmaf(delta,1.f/(j*4+i+1),v.mean);
            v.m2=fmaf(delta,f-v.mean,v.m2);
        }
    }
    float count=12.f;
    #pragma unroll
    for(int stride=16;stride>0;stride/=2) {
        Moments other{__shfl_down_sync(0xffffffff,v.mean,stride),__shfl_down_sync(0xffffffff,v.m2,stride)};
        v=combine(v,other,count);
        count*=2.f;
    }
    if(!lane) warps[warp]=v;
    __syncthreads();
    if(tid==0) {
        auto a=combine(warps[0],warps[2],384.f), b=combine(warps[1],warps[3],384.f);
        auto final=combine(a,b,768.f);
        warps[0]={final.mean,rsqrtf(__fdiv_rn(final.m2,1536.f)+eps)};
    }
    __syncthreads();
    int frame=row/frame_len;
    #pragma unroll
    for(int j=0;j<3;j++) {
        int col=j*128+tid;
        Four scale=s[frame*ss+col],shift=t[frame*ts+col],y;
        #pragma unroll
        for(int i=0;i<4;i++) {
            float norm=(__bfloat162float(input[j].x[i])-warps[0].mean)*warps[0].m2;
            if (stage==Stage::GatedAffine) {
                y.x[i]=__float2bfloat16_rn(fmaf(norm,__bfloat162float(scale.x[i]),__bfloat162float(shift.x[i])));
                continue;
            }
            BF z=__float2bfloat16_rn(norm);
            BF a=__float2bfloat16_rn(1.f+__bfloat162float(scale.x[i]));
            BF p=__float2bfloat16_rn(__bfloat162float(z)*__bfloat162float(a));
            y.x[i]=__float2bfloat16_rn(__bfloat162float(p)+__bfloat162float(shift.x[i]));
        }
        out[row*384+col]=y;
    }
}
extern "C" void layernorm_mod(const Four* x,const Four* s,const Four* t,Four* out,int n,float eps,int frame_len,int ss,int ts,cudaStream_t stream) {
    normalize_modulate<Stage::Modulate><<<n,128,0,stream>>>(x,s,t,out,eps,frame_len,ss/4,ts/4,nullptr,nullptr,nullptr,0);
}
extern "C" void residual_norm_mod(const Four* x,const Four* addend,const Four* s,const Four* t,Four* sum,Four* out,
                                  int n,float eps,int frame_len,int ss,int ts,cudaStream_t stream) {
    normalize_modulate<Stage::AddModulate><<<n,128,0,stream>>>(x,s,t,out,eps,frame_len,ss/4,ts/4,addend,sum,nullptr,0);
}
extern "C" void gated_residual_norm(const Four* x,const Four* addend,const Four* gate,const Four* w,const Four* bias,
                                    Four* sum,Four* out,int n,float eps,int frame_len,int gs,cudaStream_t stream) {
    normalize_modulate<Stage::GatedAffine><<<n,128,0,stream>>>(x,w,bias,out,eps,frame_len,0,0,addend,sum,gate,gs/4);
}
