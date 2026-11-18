#include <cute/tensor.hpp>
#include <ATen/native/transformers/cuda/flash_attn/flash.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_fwd_kernel.h>
using namespace pytorch_flash;
using Traits=Flash_fwd_kernel_traits<128,128,64,4,false,false,cutlass::bfloat16_t>;
template<bool Compress>
__global__ void ctx(Flash_fwd_params p) {
    compute_attn_1rowblock<Traits,false,false,false,false,true,true,false,Compress>(p,0,blockIdx.y,blockIdx.x);
}
extern "C" int run_ctx(void *q,void *k,void *v,void *o,void *lse,int compressed,cudaStream_t stream) {
    Flash_fwd_params p{};
    p.q_ptr=q;p.k_ptr=k;p.v_ptr=v;p.o_ptr=o;p.softmax_lse_ptr=lse;
    p.b=1;p.d=p.d_rounded=128;p.h=p.h_k=12;p.h_h_k_ratio=1;
    p.seqlen_q=p.seqlen_q_rounded=4608;p.seqlen_k=p.seqlen_k_rounded=512;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=1536;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128;
    p.scale_softmax=1.f/sqrtf(128.f);p.scale_softmax_log2=p.scale_softmax*float(M_LOG2E);
    p.is_seqlens_k_cumulative=true;p.p_dropout=p.rp_dropout=1.f;
    auto kernel=compressed?ctx<true>:ctx<false>;
    constexpr int smem=Traits::kSmemSize;
    cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
    kernel<<<dim3(36,12),128,smem,stream>>>(p);
    return cudaGetLastError();
}
