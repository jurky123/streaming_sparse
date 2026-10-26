"""One-pass FP64 rotary arithmetic, preserving the backbone's BF16 rounding."""
import torch
POSITION=None
import triton as tr
import triton.language as tl

@tr.jit
def _rope(X,F,O,N:tl.constexpr,H:tl.constexpr,HEIGHT:tl.constexpr,WIDTH:tl.constexpr,START,
          XS:tl.constexpr,XH:tl.constexpr,BLOCK:tl.constexpr):
 z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);pair=z%64;head=z//64%H;token=z//(64*H)
 frame=token//(HEIGHT*WIDTH)+tl.load(START);y=token//WIDTH%HEIGHT;x=token%WIDTH
 pos=tl.where(pair<22,frame,tl.where(pair<43,y,x));fptr=(pos*64+pair)*2
 fr=tl.load(F+fptr,token<N,0.);fi=tl.load(F+fptr+1,token<N,0.)
 xp=token*XS+head*XH+pair*2
 a=tl.load(X+xp,token<N,0.).to(tl.float64);b=tl.load(X+xp+1,token<N,0.).to(tl.float64)
 real=a*fr-b*fi;imag=a*fi+b*fr
 tl.store(O+2*z,real.to(tl.float32),token<N);tl.store(O+2*z+1,imag.to(tl.float32),token<N)

def rope(x,grid,freqs,start_frame=0):
 assert x.shape[0]==1 and x.shape[-1]==128 and x.dtype==torch.bfloat16
 f,h,w=grid[0].tolist();n=f*h*w;assert x.shape[1]==n
 out=torch.empty(x.shape,device=x.device,dtype=x.dtype)
 _rope[(tr.cdiv(n*x.shape[2]*64,256),)](x,freqs.view(torch.float64),out,n,x.shape[2],h,w,POSITION,x.stride(1),x.stride(2),256,num_warps=4,enable_fp_fusion=False)
 return out
