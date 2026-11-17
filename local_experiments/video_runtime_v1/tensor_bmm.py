"""FP32-input batched GEMM using three TF32 products; no input BF16 rounding."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def kernel(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,
           B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,
           BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32,BF16_INPUTS:tl.constexpr=False):
    x=tl.program_id(0);y=tl.program_id(1);h=tl.program_id(2)
    m=x*BM+tl.arange(0,BM);n=y*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for off in range(tr.cdiv(K,BK)):
        kk=off*BK+k
        a=tl.load(A+h*A0+m[:,None]*A1+kk[None,:]*A2,(m[:,None]<M)&(kk[None,:]<K),0)
        b=tl.load(B+h*B0+kk[:,None]*B1+n[None,:]*B2,(kk[:,None]<K)&(n[None,:]<N),0)
        if BF16_INPUTS:
            acc+=tl.dot(a.to(tl.bfloat16),b.to(tl.bfloat16))
        else:
            acc+=tl.dot(a,b,input_precision='tf32x3')
    tl.store(C+h*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@torch.library.custom_op('streaming_sparse::tensor_bmm',mutates_args=())
def tensor_bmm(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    h,m,k=a.shape;n=b.shape[-1]
    c=torch.empty((h,m,n),device=a.device,dtype=torch.float32)
    kernel[(tr.cdiv(m,32),tr.cdiv(n,64),h)](a,b,c,m,n,k,*a.stride(),*b.stride(),num_warps=4)
    return c

@tensor_bmm.register_fake
def _(a,b):return a.new_empty((a.shape[0],a.shape[1],b.shape[2]),dtype=torch.float32)


@torch.library.custom_op('streaming_sparse::bf16_inputs_bmm',mutates_args=())
def bf16_inputs_bmm(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    """Requires values exactly representable in BF16, as in deployed Q and R4 K."""
    h,m,k=a.shape;n=b.shape[-1]
    c=torch.empty((h,m,n),device=a.device,dtype=torch.float32)
    kernel[(tr.cdiv(m,32),tr.cdiv(n,64),h)](a,b,c,m,n,k,*a.stride(),*b.stride(),BF16_INPUTS=True,num_warps=4)
    return c

@bf16_inputs_bmm.register_fake
def _(a,b):return a.new_empty((a.shape[0],a.shape[1],b.shape[2]),dtype=torch.float32)
