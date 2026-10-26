// Adapter for SpargeAttn's Apache-2.0 SM80 kernel. Original kernel attribution
// and license are retained in sparge_kernel.cuh and the vendored repository.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "sparge_kernel.cuh"
template<int G>
void launch(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor o,torch::Tensor lut,torch::Tensor count,torch::Tensor qs,torch::Tensor ks,int pages,int tail){
 auto kernel=qk_int_sv_f16_block_sparse_attn_kernel<G,64,16,64,128,DataType::kInt8,QuantGranularity::kPerBlock,QuantGranularity::kPerBlock,true,PVThresholdMode::kNone,nv_bfloat16,ComputeUnit::kTensorCore,MaskMode::kNone,false>;
 size_t smem=std::max(G*128+64*128+64*128*2,G*128*2);
 C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 dim3 grid((q.size(1)+G-1)/G,q.size(0),1);dim3 block(32,G/16);
 kernel<<<grid,block,smem,at::cuda::getCurrentCUDAStream()>>>(
 q.data_ptr<int8_t>(),k.data_ptr<int8_t>(),reinterpret_cast<half*>(v.data_ptr()),reinterpret_cast<nv_bfloat16*>(o.data_ptr()),
 nullptr,lut.data_ptr<int32_t>(),count.data_ptr<int32_t>(),nullptr,qs.data_ptr<float>(),ks.data_ptr<float>(),
 q.size(1),k.size(1),1,
 q.numel(),q.stride(1),q.stride(0),k.numel(),k.stride(1),k.stride(0),v.numel(),v.stride(1),v.stride(0),o.numel(),o.stride(1),o.stride(0),0.08838834764831845f,pages,tail);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor o,torch::Tensor lut,torch::Tensor count,torch::Tensor qs,torch::Tensor ks,int group,int pages,int tail){
 TORCH_CHECK(q.is_cuda()&&q.is_contiguous()&&k.is_contiguous()&&v.is_contiguous()&&lut.is_contiguous());
 TORCH_CHECK(q.scalar_type()==at::kChar&&k.scalar_type()==at::kChar&&v.scalar_type()==at::kHalf&&o.scalar_type()==at::kBFloat16);
 TORCH_CHECK(q.size(2)==128&&lut.size(2)==k.size(1)/64);
 if(group==32)launch<32>(q,k,v,o,lut,count,qs,ks,pages,tail);
 else if(group==64)launch<64>(q,k,v,o,lut,count,qs,ks,pages,tail);
 else if(group==128)launch<128>(q,k,v,o,lut,count,qs,ks,pages,tail);
 else TORCH_CHECK(false,"Unsupported group");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
