"""Single-input FP64 RoPE for the native causal-attention entry."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _rope_single(X,F,O,N:tl.constexpr,H:tl.constexpr,GH:tl.constexpr,GW:tl.constexpr,START,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N*H*64;token=i//(H*64);pair=i%64
 row=tl.where(pair<22,token//(GH*GW)+START,tl.where(pair<43,(token//GW)%GH,token%GW))
 real=tl.load(F+(row*64+pair)*2,ok,0);imag=tl.load(F+(row*64+pair)*2+1,ok,0)
 offset=(i//64)*128+pair*2
 a=tl.load(X+offset,ok,0).to(tl.float64);b=tl.load(X+offset+1,ok,0).to(tl.float64)
 tl.store(O+offset,(a*real-b*imag).to(tl.float32),ok);tl.store(O+offset+1,(a*imag+b*real).to(tl.float32),ok)

def apply(x,grid_sizes,freqs,start_frame=0):
 assert x.is_contiguous() and x.dtype==torch.bfloat16 and x.shape[0]==1 and x.shape[-2:]==(12,128)
 f,h,w=grid_sizes[0].tolist();n=x.shape[1];assert f*h*w==n and start_frame+f<=freqs.shape[0]
 out=torch.empty_like(x)
 _rope_single[(tr.cdiv(n*12*64,256),)](x,torch.view_as_real(freqs),out,n,12,h,w,start_frame,256,num_warps=4,enable_fp_fusion=False)
 return out

class NativeRope:
 def __init__(self):
  import wan.modules.causal_model as causal
  self.module=causal;self.original=causal.causal_rope_apply;causal.causal_rope_apply=apply
 def close(self):self.module.causal_rope_apply=self.original
