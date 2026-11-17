"""Positive FP64 exclusion softmax and value reduction without weight materialization."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
ROWS=4
WARPS=4
KERNEL_STATS={}
@tr.jit
def kernel(L,V,LO,O,N:tl.constexpr,K:tl.constexpr,D:tl.constexpr,ROW:tl.constexpr,LS0:tl.constexpr,LS1:tl.constexpr,LS2:tl.constexpr,VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,VS3:tl.constexpr,M:tl.constexpr):
    n=tl.program_id(0);b=n//M;m=n%M
    row=tl.program_id(1)*ROW+tl.arange(0,ROW);k=tl.arange(0,K);d=tl.arange(0,D)
    raw=tl.load(L+b*LS0+m*LS1+k*LS2)
    logs=tl.where(row[:,None]!=k[None,:],raw[None,:],-float('inf'))
    mx=tl.max(logs,1);safe=tl.where(tl.abs(mx)==float('inf'),0.,mx)
    w=lib.exp(logs-safe[:,None]);z=tl.sum(w,1)
    weights=w/z[:,None]
    v=tl.load(V+b*VS0+m*VS1+k[:,None]*VS2+d[None,:]*VS3)
    out=tl.sum(weights[:,:,None]*v[None,:,:],1)
    tl.store(LO+n*K+row,safe+lib.log(z),row<K)
    tl.store(O+(n*K+row[:,None])*D+d[None,:],out,row[:,None]<K)
@torch.library.custom_op('streaming_sparse::exclude_fused',mutates_args=())
def exclude_fused(logs:torch.Tensor,values:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    b,m,k=logs.shape;d=values.shape[-1];lo=torch.empty_like(logs);out=torch.empty_like(values,memory_format=torch.contiguous_format)
    compiled=kernel[(b*m,tr.cdiv(k,ROWS))](logs,values,lo,out,b*m,k,d,ROWS,*logs.stride(),*values.stride(),m,num_warps=WARPS,enable_fp_fusion=False)
    KERNEL_STATS[k]=dict(registers=compiled.n_regs,spills=compiled.n_spills,rows=ROWS,warps=WARPS)
    return lo,out
@exclude_fused.register_fake
def _(logs,values):return torch.empty_like(logs),torch.empty_like(values,memory_format=torch.contiguous_format)
