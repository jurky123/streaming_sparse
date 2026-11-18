"""Preserve PyTorch's FP32 mean/rsqrt; remove RMSNorm pointwise temporaries."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _square(X,S,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);x=tl.load(X+i,i<N,0).to(tl.float32)
 tl.store(S+i,x*x,i<N)

@tr.jit
def _finish(X,I,W,O,N:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
 x=tl.load(X+i,ok,0).to(tl.float32);inv=tl.load(I+i//D,ok,0)
 normalized=(x*inv).to(X.dtype.element_ty).to(tl.float32)
 weight=tl.load(W+i%D,ok,0).to(tl.float32)
 tl.store(O+i,normalized*weight,ok)

def reference(x,weight,eps):
 f=x.float();return (f*torch.rsqrt(f.pow(2).mean(-1,keepdim=True)+eps)).to(x.dtype)*weight

def fused(x,weight,eps):
 assert x.is_contiguous() and x.dtype==weight.dtype==torch.bfloat16
 squared=torch.empty(x.shape,device=x.device,dtype=torch.float32)
 _square[(tr.cdiv(x.numel(),1024),)](x,squared,x.numel(),1024,num_warps=4,enable_fp_fusion=False)
 inv=torch.rsqrt(squared.mean(-1,keepdim=True)+eps)
 out=torch.empty_like(x)
 _finish[(tr.cdiv(x.numel(),1024),)](x,inv,weight,out,x.numel(),x.shape[-1],1024,num_warps=4,enable_fp_fusion=False)
 return out

class RMSDataflow:
 def __init__(self,engine,optimized=True,observe=None,profile=False,mean_kernel=False):
  from wan.modules.model import WanRMSNorm
  self.function=fused
  if mean_kernel:
   from .rms_mean import fused as mean_fused
   self.function=mean_fused
  self.original=[];self.observe=observe;self.optimized=optimized;self.profile=profile
  for name,module in engine.wrapper.model.named_modules():
   if isinstance(module,WanRMSNorm):
    old=module.forward;self.original.append((module,old))
    def forward(x,label=name,m=module,original=old):
     fn=lambda:self.function(x,m.weight,m.eps) if self.optimized else original(x)
     if self.profile:
      with torch.profiler.record_function('rms.'+label):out=fn()
     else:out=fn()
     if self.observe is not None:self.observe(label,x,m.weight,m.eps)
     return out
    module.forward=forward
 def close(self):
  for module,fn in self.original:module.forward=fn
