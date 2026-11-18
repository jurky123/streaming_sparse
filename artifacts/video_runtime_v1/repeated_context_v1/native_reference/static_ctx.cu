#include <cute/tensor.hpp>
#include <ATen/native/transformers/cuda/flash_attn/flash.h>
#include <ATen/native/transformers/cuda/flash_attn/flash_fwd_kernel.h>
using namespace pytorch_flash;
template<int Rows,int Warps,bool Compress>
__global__ void specialized(void *q,void *k,void *v,void *o,void *lse) {
    using Traits=Flash_fwd_kernel_traits<128,Rows,64,Warps,false,false,cutlass::bfloat16_t>;
    Flash_fwd_params p{};
    p.q_ptr=q;p.k_ptr=k;p.v_ptr=v;p.o_ptr=o;p.softmax_lse_ptr=lse;
    p.b=1;p.d=p.d_rounded=128;p.h=p.h_k=12;p.h_h_k_ratio=1;
    p.seqlen_q=p.seqlen_q_rounded=4608;p.seqlen_k=p.seqlen_k_rounded=512;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=1536;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128;
    p.scale_softmax=1.f/sqrtf(128.f);p.scale_softmax_log2=p.scale_softmax*float(M_LOG2E);
    p.is_seqlens_k_cumulative=true;p.p_dropout=p.rp_dropout=1.f;
    compute_attn_1rowblock<Traits,false,false,false,false,true,true,false,Compress>(p,0,blockIdx.y,blockIdx.x);
}
template<int Rows,int Warps>
int launch(void *q,void *k,void *v,void *o,void *lse,int compress,cudaStream_t stream) {
    using Traits=Flash_fwd_kernel_traits<128,Rows,64,Warps,false,false,cutlass::bfloat16_t>;
    auto kernel=compress?specialized<Rows,Warps,true>:specialized<Rows,Warps,false>;
    constexpr int smem=Traits::kSmemSize;
    cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
    kernel<<<dim3(4608/Rows,12),32*Warps,smem,stream>>>(q,k,v,o,lse);
    return cudaGetLastError();
}
extern "C" int run_ctx(void *q,void *k,void *v,void *o,void *lse,int compressed,cudaStream_t stream) {
    return launch<ROWS,WARPS>(q,k,v,o,lse,compressed,stream);
}
