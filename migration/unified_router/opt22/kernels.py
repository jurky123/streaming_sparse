"""Fused BF16 modulation/residual and RMSNorm with explicit rounding boundaries."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _pointwise(X,Y,S,B,O,N:tl.constexpr,D:tl.constexpr,TOKENS:tl.constexpr,SS:tl.constexpr,BS:tl.constexpr,
               MODE:tl.constexpr,BLOCK:tl.constexpr,XS:tl.constexpr,XD:tl.constexpr,YS:tl.constexpr,YD:tl.constexpr):
 z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);c=z%D;frame=z//D//TOKENS
 x=tl.load(X+(z//D)*XS+c*XD,z<N,0.).to(tl.float32);s=tl.load(S+frame*SS+c,z<N,0.).to(tl.float32)
 if MODE==0:
  b=tl.load(B+frame*BS+c,z<N,0.).to(tl.float32)
  scale=(s+1.).to(tl.bfloat16).to(tl.float32)
  y=(x*scale).to(tl.bfloat16).to(tl.float32)+b
 else:
  a=tl.load(Y+(z//D)*YS+c*YD,z<N,0.).to(tl.float32)
  y=x+(a*s).to(tl.bfloat16).to(tl.float32)
 tl.store(O+z,y,z<N)

def modulate(x,scale,bias):
 assert x.is_contiguous() and x.dtype==scale.dtype==bias.dtype==torch.bfloat16
 out=torch.empty_like(x);n=x.numel();d=x.shape[-1];tokens=x.shape[1]//scale.shape[1]
 _pointwise[(tr.cdiv(n,512),)](x,x,scale,bias,out,n,d,tokens,scale.stride(1),bias.stride(1),0,512,x.stride(1),x.stride(2),x.stride(1),x.stride(2),num_warps=4,enable_fp_fusion=False)
 return out

def residual(x,branch,gate):
 assert x.dtype==branch.dtype==gate.dtype==torch.bfloat16,(x.dtype,branch.dtype,gate.dtype)
 out=torch.empty(x.shape,device=x.device,dtype=x.dtype);n=x.numel();d=x.shape[-1];tokens=x.shape[1]//gate.shape[1]
 _pointwise[(tr.cdiv(n,512),)](x,branch,gate,gate,out,n,d,tokens,gate.stride(1),gate.stride(1),1,512,x.stride(1),x.stride(2),branch.stride(1),branch.stride(2),num_warps=4,enable_fp_fusion=False)
 return out

@tr.jit
def _rms_apply(X,M,W,O,N:tl.constexpr,D:tl.constexpr,EPS:tl.constexpr,XS:tl.constexpr,BLOCK:tl.constexpr):
 z=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=z//D;c=z%D
 x=tl.load(X+row*XS+c,z<N,0.).to(tl.float32);mean=tl.load(M+row,z<N,0.)
 r=tl.rsqrt(mean+EPS);weight=tl.load(W+c,z<N,0.).to(tl.float32)
 y=(x*r).to(tl.bfloat16).to(tl.float32)*weight
 tl.store(O+z,y,z<N)

@tr.jit
def _rms(X,W,O,D:tl.constexpr,EPS:tl.constexpr,XS:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);c=tl.arange(0,BLOCK)
 x=tl.load(X+row*XS+c,c<D,0.).to(tl.float32)
 mean=tl.sum(x*x,0)*(1./D);r=tl.rsqrt(mean+EPS)
 w=tl.load(W+c,c<D,0.).to(tl.float32)
 y=(x*r).to(tl.bfloat16).to(tl.float32)*w
 tl.store(O+row*D+c,y,c<D)

def rms(x,weight,eps,mode='native_stats'):
 assert x.dtype==weight.dtype==torch.bfloat16 and x.stride(-1)==1
 out=torch.empty(x.shape,device=x.device,dtype=x.dtype);d=x.shape[-1]
 if mode=='native_stats':
  mean=x.float().pow(2).mean(dim=-1,keepdim=True)
  _rms_apply[(tr.cdiv(x.numel(),512),)](x,mean,weight,out,x.numel(),d,eps,x.stride(-2),512,num_warps=4,enable_fp_fusion=False)
 elif mode=='fused':_rms[(x.numel()//d,)](x,weight,out,d,eps,x.stride(-2),tr.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
 else:raise ValueError(mode)
 return out

@tr.jit
def _rms_ordered(X,W,O,D:tl.constexpr,EPS:tl.constexpr,XS:tl.constexpr,BLOCK:tl.constexpr):
 # Match the installed PyTorch Reduce.cuh input-vectorized mean:
 # 32 lanes, four independent accumulators, then ascending shuffle offsets.
 row=tl.program_id(0);lane=tl.arange(0,32)
 v0=tl.zeros((32,),tl.float32);v1=v0;v2=v0;v3=v0
 for k in range(D//128):
  c=(k*32+lane)*4
  a=tl.load(X+row*XS+c).to(tl.float32);b=tl.load(X+row*XS+c+1).to(tl.float32)
  e=tl.load(X+row*XS+c+2).to(tl.float32);f=tl.load(X+row*XS+c+3).to(tl.float32)
  v0=v0+a*a;v1=v1+b*b;v2=v2+e*e;v3=v3+f*f
 total=((v0+v1)+v2)+v3
 for offset in tl.static_range(0,5):
  other=tl.inline_asm_elementwise("shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",constraints="=f,f,r",args=[total,1<<offset],dtype=tl.float32,is_pure=True,pack=1)
  total=total+other
 mean=tl.sum(tl.where(lane==0,total,0.),0)*(1./D);r=tl.rsqrt(mean+EPS)
 c=tl.arange(0,BLOCK);x=tl.load(X+row*XS+c,c<D,0.).to(tl.float32);w=tl.load(W+c,c<D,0.).to(tl.float32)
 y=(x*r).to(tl.bfloat16).to(tl.float32)*w
 tl.store(O+row*D+c,y,c<D)

def rms_ordered(x,weight,eps):
 d=x.shape[-1];assert d==1536 and x.numel()//d>=16 and x.dtype==weight.dtype==torch.bfloat16
 out=torch.empty(x.shape,device=x.device,dtype=x.dtype)
 _rms_ordered[(x.numel()//d,)](x,weight,out,d,eps,x.stride(-2),tr.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
 return out
