"""Positive prefix/suffix exclusion aggregates, with dominant-page rescaling."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
GROUP=4
WARPS=4
KERNEL_STATS={}
@tr.jit
def scan_kernel(L,V,LO,O,N:tl.constexpr,K:tl.constexpr,D:tl.constexpr,G:tl.constexpr,LS0:tl.constexpr,LS1:tl.constexpr,LS2:tl.constexpr,VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,VS3:tl.constexpr,M:tl.constexpr):
    n=tl.program_id(0)*G+tl.arange(0,G);b=n//M;m=n%M
    k=tl.arange(0,K);d=tl.arange(0,D)
    logs=tl.load(L+b[:,None]*LS0+m[:,None]*LS1+k[None,:]*LS2,n[:,None]<N,0)
    value=tl.load(V+b[:,None,None]*VS0+m[:,None,None]*VS1+k[None,:,None]*VS2+d[None,None,:]*VS3,n[:,None,None]<N,0)
    top=tl.argmax(logs,1);maximum=tl.max(logs,1)
    ll=tl.load(L+b[:,None]*LS0+m[:,None]*LS1+(k[None,:]-1)*LS2,(n[:,None]<N)&(k[None,:]>0),-float('inf'))
    rl=tl.load(L+b[:,None]*LS0+m[:,None]*LS1+(k[None,:]+1)*LS2,(n[:,None]<N)&(k[None,:]<K-1),-float('inf'))
    wl=lib.exp(ll-maximum[:,None]);wr=lib.exp(rl-maximum[:,None])
    vl=tl.load(V+b[:,None,None]*VS0+m[:,None,None]*VS1+(k[None,:,None]-1)*VS2+d[None,None,:]*VS3,(n[:,None,None]<N)&(k[None,:,None]>0),0)
    vr=tl.load(V+b[:,None,None]*VS0+m[:,None,None]*VS1+(k[None,:,None]+1)*VS2+d[None,None,:]*VS3,(n[:,None,None]<N)&(k[None,:,None]<K-1),0)
    z=tl.associative_scan(wl,1,_sum)+tl.associative_scan(wr,1,_sum,reverse=True)
    num=tl.associative_scan(wl[:,:,None]*vl,1,_sum)+tl.associative_scan(wr[:,:,None]*vr,1,_sum,reverse=True)
    # The one exclusion which removes the maximum uses its own scale, so a
    # 1000-log-unit dominant page never destroys the remaining positive mass.
    other=tl.where(k[None,:]!=top[:,None],logs,-float('inf'))
    second=tl.max(other,1);wd=lib.exp(other-second[:,None]);zd=tl.sum(wd,1)
    vd=tl.sum(wd[:,:,None]*value,1)/zd[:,None]
    dominant=k[None,:]==top[:,None]
    lo=tl.where(dominant,second[:,None]+lib.log(zd)[:,None],maximum[:,None]+lib.log(z))
    out=tl.where(dominant[:,:,None],vd[:,None,:],num/z[:,:,None])
    tl.store(LO+n[:,None]*K+k[None,:],lo,n[:,None]<N)
    tl.store(O+(n[:,None,None]*K+k[None,:,None])*D+d[None,None,:],out,n[:,None,None]<N)
@tr.jit
def _sum(a,b):return a+b
@torch.library.custom_op('streaming_sparse::exclude_scan',mutates_args=())
def exclude_scan(logs:torch.Tensor,values:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    b,m,k=logs.shape;d=values.shape[-1]
    lo=torch.empty(logs.shape,device=logs.device,dtype=logs.dtype);out=torch.empty(values.shape,device=values.device,dtype=values.dtype)
    compiled=scan_kernel[(tr.cdiv(b*m,GROUP),)](logs,values,lo,out,b*m,k,d,GROUP,*logs.stride(),*values.stride(),m,num_warps=WARPS,enable_fp_fusion=False)
    KERNEL_STATS[k]=dict(registers=compiled.n_regs,spills=compiled.n_spills,group=GROUP,warps=WARPS)
    return lo,out
@exclude_scan.register_fake
def _(logs,values):return logs.new_empty(logs.shape),values.new_empty(values.shape)
