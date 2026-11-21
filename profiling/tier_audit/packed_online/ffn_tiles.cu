// Module probe for SM80. The BF16 table contains PyTorch GELU for all 65536 input bit patterns.
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include <cuda_bf16.h>

using BF = cutlass::bfloat16_t;
__device__ const BF *gelu_table;
extern "C" int set_gelu_table(const BF *p) {
    return cudaMemcpyToSymbol(gelu_table, &p, sizeof(p));
}

template<class T> struct RoundedGelu;
template<> struct RoundedGelu<float> {
    CUTLASS_DEVICE float operator()(float x) const {
        return float(gelu_table[BF(x).raw()]);
    }
};
template<int N> struct RoundedGelu<cutlass::Array<float,N>> {
    CUTLASS_DEVICE cutlass::Array<float,N> operator()(cutlass::Array<float,N> const &x) const {
        cutlass::Array<float,N> out;
        CUTLASS_PRAGMA_UNROLL
        for (int i=0;i<N;++i) out[i]=RoundedGelu<float>()(x[i]);
        return out;
    }
};
using Row=cutlass::layout::RowMajor;
using Col=cutlass::layout::ColumnMajor;
using Epilogue=cutlass::epilogue::thread::LinearCombinationGeneric<RoundedGelu,BF,8,float,float,cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
template<int BM,int BN,int BK,int ST>
int launch(int m,int n,int k,const BF *x,const BF *w,const BF *bias,BF *out,cudaStream_t stream) {
    using Op=cutlass::gemm::device::Gemm<BF,Row,BF,Col,BF,Row,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<BM,BN,BK>,cutlass::gemm::GemmShape<BM/2,BN/2,BK>,cutlass::gemm::GemmShape<16,8,16>,
        Epilogue,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,ST>;
    typename Op::Arguments args({m,n,k},{x,k},{w,k},{bias,0},{out,n},{1.f,1.f});
    return int(Op()(args,nullptr,stream));
}
extern "C" int ffn_gelu(int bn,int m,int n,int k,const BF *x,const BF *w,const BF *bias,BF *out,cudaStream_t stream) {
    if(bn==0)return launch<128,128,32,3>(m,n,k,x,w,bias,out,stream);
    if(bn==1)return launch<128,256,32,3>(m,n,k,x,w,bias,out,stream);
    if(bn==2)return launch<128,128,64,3>(m,n,k,x,w,bias,out,stream);
    if(bn==3)return launch<128,128,64,4>(m,n,k,x,w,bias,out,stream);
    if(bn==4)return launch<128,256,64,3>(m,n,k,x,w,bias,out,stream);
    if(bn==5)return launch<64,128,64,4>(m,n,k,x,w,bias,out,stream);
    if(bn==6)return launch<128,64,64,4>(m,n,k,x,w,bias,out,stream);
    if(bn==7)return launch<64,256,64,3>(m,n,k,x,w,bias,out,stream);
    if(bn==8)return launch<128,128,32,5>(m,n,k,x,w,bias,out,stream);
    return -1;
}
