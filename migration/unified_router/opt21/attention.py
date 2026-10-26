"""Shared-page attention with tiled split merge and fused query restore."""
import torch
import triton as tr
import triton.language as tl
from ..opt19.attention import _shared

@tr.jit
def _merge_restore(P,L,IX,O,N:tl.constexpr,S:tl.constexpr,R:tl.constexpr,RESTORE:tl.constexpr):
 h=tl.program_id(1);r=tl.program_id(0)*R+tl.arange(0,R);d=tl.arange(0,128);s=tl.arange(0,S)
 lse=tl.load(L+(h*N+r[:,None])*S+s[None,:],r[:,None]<N,0.)
 maximum=tl.max(lse,1);weights=tl.exp2(lse-maximum[:,None]);denom=tl.sum(weights,1)
 partial=tl.load(P+((h*N+r[:,None,None])*S+s[None,:,None])*128+d[None,None,:],r[:,None,None]<N,0.)
 y=tl.sum(partial*weights[:,:,None],1)/denom[:,None]
 target=tl.load(IX+h*N+r,r<N,0) if RESTORE else r
 tl.store(O+(h*N+target[:,None])*128+d[None,:],y,r[:,None]<N)

def attention(q,kv,ids,quant,ix=None,splits=2,warps=2,stages=2,rows=8):
 h,n,d=q.shape;b=kv.state['key'].shape[1];out=torch.empty_like(q)
 partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32)
 lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
 _shared[(tr.cdiv(n,32),h,splits)](q,quant.k,kv.v,quant.scale,ids,partial,lse,n,kv.k.shape[1],b,32,ids.shape[1],ids.shape[2],splits,True,num_warps=warps,num_stages=stages)
 assert splits>1
 _merge_restore[(tr.cdiv(n,rows),h)](partial,lse,ix if ix is not None else ids,out,n,splits,rows,ix is not None,num_warps=4,enable_fp_fusion=False)
 return out

from triton.language.extra.cuda.libdevice import nearbyint

@tr.jit
def _subtile(Q,K,V,KS,IDS,O,LSE,N:tl.constexpr,NK:tl.constexpr,B:tl.constexpr,T:tl.constexpr,KEEP:tl.constexpr,
             SPLITS:tl.constexpr,G:tl.constexpr,BN:tl.constexpr):
 tile=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2)
 qi=tile*G+tl.arange(0,G);d=tl.arange(0,128);t=tl.arange(0,BN)
 parent=tile//(32//G)
 # Preserve quantization across the original 32-query selection tile.
 qi32=parent*32+tl.arange(0,32)
 qall=tl.load(Q+(h*N+qi32[:,None])*128+d[None,:],qi32[:,None]<N,0).to(tl.float32)
 qs=tl.maximum(tl.max(tl.max(tl.abs(qall),1),0)/127.,1e-10)
 q=tl.load(Q+(h*N+qi[:,None])*128+d[None,:],qi[:,None]<N,0).to(tl.float32)
 q=nearbyint(q/qs).to(tl.int8)
 maximum=tl.full((G,),float('-inf'),tl.float32);denom=tl.zeros((G,),tl.float32);acc=tl.zeros((G,128),tl.float32)
 per=tr.cdiv(KEEP,SPLITS);start=split*per;end=tl.minimum((split+1)*per,KEEP)
 for j in range(start,end,BN//64):
  which=j+t//64;page=tl.load(IDS+(h*T+parent)*KEEP+which,which<end,0)
  ki=page//74*4680+page%74*64+t%64;valid=(which<end)&(ki<NK)&(page%74*64+t%64<4680)
  k=tl.load(K+(h*NK+ki[None,:])*128+d[:,None],valid[None,:],0)
  scale=tl.load(KS+h*B+page)
  logits=tl.dot(q,k).to(tl.float32)*(qs*scale[None,:])*0.12751743082459617
  logits=tl.where(valid[None,:],logits,float('-inf'));newmax=tl.maximum(maximum,tl.max(logits,1))
  alpha=tl.exp2(maximum-newmax);prob=tl.exp2(logits-newmax[:,None]);v=tl.load(V+(h*NK+ki[:,None])*128+d[None,:],valid[:,None],0)
  acc=acc*alpha[:,None];acc=tl.dot(prob.to(v.dtype),v,acc);denom=denom*alpha+tl.sum(prob,1);maximum=newmax
 tl.store(O+((h*N+qi[:,None])*SPLITS+split)*128+d[None,:],acc/denom[:,None],qi[:,None]<N)
 tl.store(LSE+(h*N+qi)*SPLITS+split,maximum+tl.log2(denom),qi<N)

def subtile(q,kv,ids,quant,ix=None,splits=2,warps=4,stages=2,group=16,bn=64):
 h,n,d=q.shape;b=kv.state['key'].shape[1];out=torch.empty_like(q)
 partial=torch.empty((h,n,splits,128),device=q.device,dtype=torch.float32);lse=torch.empty((h,n,splits),device=q.device,dtype=torch.float32)
 _subtile[(tr.cdiv(n,group),h,splits)](q,quant.k,kv.v,quant.scale,ids,partial,lse,n,kv.k.shape[1],b,ids.shape[1],ids.shape[2],splits,group,bn,num_warps=warps,num_stages=stages)
 _merge_restore[(tr.cdiv(n,8),h)](partial,lse,ix if ix is not None else ids,out,n,splits,8,ix is not None,num_warps=4)
 return out
