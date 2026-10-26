"""Local FP16 Tensor Core projection; global/backbone TF32 stays disabled."""
import torch,triton as tr
import triton.language as tl
@tr.jit
def _project(Q,B,O,N:tl.constexpr,QH:tl.constexpr,QN:tl.constexpr,BH:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr):
 h=tl.program_id(1);n=tl.program_id(0)*BM+tl.arange(0,BM);k=tl.arange(0,128);d=tl.arange(0,64)
 q=tl.load(Q+h*QH+n[:,None]*QN+k[None,:],n[:,None]<N,0.).to(tl.float16)
 basis=tl.load(B+h*BH+k[:,None]*BK+d[None,:]*BD).to(tl.float16)
 out=tl.dot(q,basis)
 tl.store(O+(h*N+n[:,None])*64+d[None,:],out,n[:,None]<N)
def project(q,basis):
 h,n,_=q.shape;out=torch.empty((h,n,64),device=q.device,dtype=torch.float16)
 _project[(tr.cdiv(n,32),h)](q,basis,out,n,q.stride(0),q.stride(1),basis.stride(0),basis.stride(1),basis.stride(2),32,num_warps=4,num_stages=1)
 return out
