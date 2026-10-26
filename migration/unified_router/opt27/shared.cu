#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
__device__ __forceinline__ float warp_max(float v){for(int x=16;x;x>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,x));return v;}
__device__ __forceinline__ float warp_sum(float v){for(int x=16;x;x>>=1)v+=__shfl_xor_sync(0xffffffff,v,x);return v;}
// Each CTA owns M queries. Scores live in shared memory until normalized and
// reduced to C page weights; only the latter reach global memory.
template<int M,int GP,int W>
__global__ void refine_kernel(const half* Q,const half* K,const float* LC,const int* IDS,float* OUT,int N,int B,int C,int G,int pages,int stride){
 int group=blockIdx.x,head=blockIdx.y,half_id=blockIdx.z,tid=threadIdx.x,warp=tid/32,lane=tid%32;
 extern __shared__ __align__(32) unsigned char storage[];
 half* qs=(half*)storage;half* ks=qs+M*64;float* logits=(float*)(ks+GP*16*64);float* scores=logits+M*GP*16;
 __shared__ int selected[GP];__shared__ int pmax;
 for(int i=tid;i<M*64;i+=W*32){int row=group*32+half_id*M+i/64;qs[i]=row<N?Q[(head*N+row)*64+i%64]:__float2half(0.f);}
 __syncthreads();
 for(int base=0;base<C;base+=GP){
  if(tid<GP){int page=base+tid<C?IDS[(head*G+group)*C+base+tid]:0;selected[tid]=page;}
  if(tid==0)pmax=2;
  __syncthreads();
  if(tid<GP && base+tid<C){int age=(B+pages-1)/pages-1-selected[tid]/pages;atomicMax(&pmax,age==0?16:age==1?8:age==2?4:2);}
  __syncthreads();int P=pmax,cols=GP*P;
  for(int i=tid;i<cols*64;i+=W*32){int proto=i/64,ci=proto/P;ks[i]=K[((head*B+selected[ci])*16+proto%P)*64+i%64];}
  __syncthreads();
  for(int tile=warp;tile<(M/16)*(cols/16);tile+=W){
   int mr=tile/(cols/16)*16,nc=tile%(cols/16)*16;
   wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
   wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
   wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
   for(int k=0;k<64;k+=16){wmma::load_matrix_sync(a,qs+mr*64+k,64);wmma::load_matrix_sync(b,ks+nc*64+k,64);wmma::mma_sync(acc,a,b,acc);}
   wmma::store_matrix_sync(logits+mr*cols+nc,acc,cols,wmma::mem_row_major);
  }
  __syncthreads();
  for(int idx=tid;idx<M*GP;idx+=W*32){int r=idx/GP,c=idx%GP;float mx=-INFINITY,vals[16];
   for(int p=0;p<P;++p){float x=fmaf(logits[r*cols+c*P+p],0.08838834764831845f,LC[(head*B+selected[c])*16+p]);vals[p]=x;mx=fmaxf(mx,x);}
   float sum=0.f;for(int p=0;p<P;++p)sum+=__expf(vals[p]-mx);
   if(base+c<C)scores[r*stride+base+c]=mx+__logf(sum);
  }
  __syncthreads();
 }
 // Each warp normalizes one query across the candidate pages.
 for(int r=warp;r<M;r+=W){float mx=-INFINITY;
  for(int c=lane;c<C;c+=32)mx=fmaxf(mx,scores[r*stride+c]);mx=warp_max(mx);float sum=0.f;
  for(int c=lane;c<C;c+=32)sum+=__expf(scores[r*stride+c]-mx);sum=warp_sum(sum);
  for(int c=lane;c<C;c+=32)scores[r*stride+c]=(group*32+half_id*M+r<N)?__expf(scores[r*stride+c]-mx)/sum:0.f;
 }
 __syncthreads();
 for(int c=warp;c<C;c+=W){float v=lane<M?scores[lane*stride+c]:0.f;v=warp_sum(v);if(lane==0)OUT[((head*G+group)*(32/M)+half_id)*C+c]=v;}
}
template<int M,int GP,int W> void launch(torch::Tensor q,torch::Tensor k,torch::Tensor lc,torch::Tensor ids,torch::Tensor out,int pages){
 int stride=(ids.size(2)+7)/8*8+8;size_t smem=M*64*2+GP*16*64*2+M*GP*16*4+M*stride*4;
 auto fn=refine_kernel<M,GP,W>;C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 fn<<<dim3(ids.size(1),q.size(0),32/M),W*32,smem,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<half*>(q.data_ptr()),reinterpret_cast<half*>(k.data_ptr()),lc.data_ptr<float>(),ids.data_ptr<int>(),out.data_ptr<float>(),q.size(1),k.size(1),ids.size(2),ids.size(1),pages,stride);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run(torch::Tensor q,torch::Tensor k,torch::Tensor lc,torch::Tensor ids,torch::Tensor out,int pages,int m,int gp,int warps){
 TORCH_CHECK(q.is_cuda()&&q.is_contiguous()&&k.is_contiguous()&&lc.is_contiguous()&&ids.is_contiguous());
 TORCH_CHECK(q.scalar_type()==at::kHalf&&k.scalar_type()==at::kHalf&&lc.scalar_type()==at::kFloat&&ids.scalar_type()==at::kInt);
 #define DISPATCH(M,GP,W) if(m==M&&gp==GP&&warps==W){launch<M,GP,W>(q,k,lc,ids,out,pages);return;}
 DISPATCH(16,8,4) DISPATCH(16,8,8) DISPATCH(16,16,4) DISPATCH(16,16,8)
 DISPATCH(32,8,4) DISPATCH(32,8,8) DISPATCH(32,16,4) DISPATCH(32,16,8)
 TORCH_CHECK(false,"Unsupported M/GP/warps");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
