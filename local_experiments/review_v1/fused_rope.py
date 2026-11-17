"""Fused Q/K rotary application retaining the original FP64 arithmetic."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _rope(Q,K,F,OQ,OK,N:tl.constexpr,H:tl.constexpr,GH:tl.constexpr,GW:tl.constexpr,START,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);active=i<N*H*64
 token=i//(H*64);pair=i%64
 row=tl.where(pair<22,token//(GH*GW)+START,tl.where(pair<43,(token//GW)%GH,token%GW))
 real=tl.load(F+(row*64+pair)*2,active,0);imag=tl.load(F+(row*64+pair)*2+1,active,0)
 offset=(i//64)*128+pair*2
 qr=tl.load(Q+offset,active,0).to(tl.float64);qi=tl.load(Q+offset+1,active,0).to(tl.float64)
 kr=tl.load(K+offset,active,0).to(tl.float64);ki=tl.load(K+offset+1,active,0).to(tl.float64)
 tl.store(OQ+offset,(qr*real-qi*imag).to(tl.float32),active);tl.store(OQ+offset+1,(qr*imag+qi*real).to(tl.float32),active)
 tl.store(OK+offset,(kr*real-ki*imag).to(tl.float32),active);tl.store(OK+offset+1,(kr*imag+ki*real).to(tl.float32),active)
@torch.no_grad()
def apply(q,k,grid_sizes,freqs,start_frame):
 assert q.shape==k.shape and q.shape[0]==1 and q.shape[-2:]==(12,128)
 assert q.dtype==k.dtype==torch.bfloat16 and q.is_contiguous() and k.is_contiguous()
 assert grid_sizes.device.type=='cpu' and grid_sizes.shape==(1,3)
 f,h,w=grid_sizes[0].tolist();n=q.shape[1];assert f*h*w==n and start_frame+f<=freqs.shape[0]
 assert freqs.dtype==torch.complex128 and freqs.shape[1]==64 and freqs.is_contiguous()
 oq=torch.empty_like(q);ok=torch.empty_like(k)
 _rope[(tr.cdiv(n*12*64,256),)](q,k,torch.view_as_real(freqs),oq,ok,n,12,h,w,start_frame,256,num_warps=4,enable_fp_fusion=False)
 return oq,ok
