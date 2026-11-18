"""FP32-input batched GEMM using three TF32 products; no input BF16 rounding."""
import torch
QK_TILE64=False
import triton as tr
import triton.language as tl

@tr.jit
def kernel(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,
           B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,
           BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32,BF16_INPUTS:tl.constexpr=False,SCALE:tl.constexpr=1., LOGIT_TILE:tl.constexpr=0):
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
    if SCALE != 1.:
        acc = acc * SCALE
    if LOGIT_TILE:
        ptr=C+((h*M+m[:,None])//16*tr.cdiv(N,LOGIT_TILE)+n[None,:]//LOGIT_TILE)*16*LOGIT_TILE
        ptr=ptr+(m[:,None]%16)*LOGIT_TILE+n[None,:]%LOGIT_TILE
    else:
        ptr=C+h*M*N+m[:,None]*N+n[None,:]
    tl.store(ptr,acc,(m[:,None]<M)&(n[None,:]<N))

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
    bm=64 if QK_TILE64 and m==1152 and k==128 else 32
    kernel[(tr.cdiv(m,bm),tr.cdiv(n,64),h)](a,b,c,m,n,k,*a.stride(),*b.stride(),BM=bm,BF16_INPUTS=True,num_warps=4)
    return c

@bf16_inputs_bmm.register_fake
def _(a,b):return a.new_empty((a.shape[0],a.shape[1],b.shape[2]),dtype=torch.float32)


@torch.library.custom_op('streaming_sparse::scaled_route_qk', mutates_args=())
def scaled_route_qk(q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    heads, rows, dim = q.shape
    columns = keys.shape[-1]
    out = torch.empty((heads, rows, columns), device=q.device, dtype=torch.float32)
    bm = 64 if QK_TILE64 and rows == 1152 and dim == 128 else 32
    kernel[(tr.cdiv(rows, bm), tr.cdiv(columns, 64), heads)](
        q, keys, out, rows, columns, dim, *q.stride(), *keys.stride(),
        BM=bm, BF16_INPUTS=True, SCALE=dim ** -0.5, num_warps=4)
    return out

@scaled_route_qk.register_fake
def _(q, keys):
    return q.new_empty((q.shape[0], q.shape[1], keys.shape[2]), dtype=torch.float32)
