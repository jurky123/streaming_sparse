"""Bitset greedy grouping directly minimizes added union pages within a window."""
import torch,triton as tr
import triton.language as tl
@tr.jit
def _or(a,b):return a|b
@tr.jit
def _popcount(x):return tl.inline_asm_elementwise('popc.b32 $0, $1;',constraints='=r,r',args=[x],dtype=tl.int32,is_pure=True,pack=1)
@tr.jit
def _pack(M,O,N:tl.constexpr,B:tl.constexpr,WORDS:tl.constexpr,P:tl.constexpr):
 h=tl.program_id(1);q=tl.program_id(0)*32+tl.arange(0,32);p=tl.arange(0,P)
 m=tl.load(M+(h*N+q[:,None])*B+p[None,:],(q[:,None]<N)&(p[None,:]<B),False)
 shifted=m.to(tl.uint32)<<(p[None,:]%32)
 bits=tl.sum(tl.reshape(shifted,(32,P//32,32)),2)
 w=tl.arange(0,P//32);tl.store(O+(h*N+q[:,None])*WORDS+w[None,:],bits,(q[:,None]<N)&(w[None,:]<WORDS))
@tr.jit
def _group(BITS,OUT,N:tl.constexpr,WORDS:tl.constexpr,P:tl.constexpr,W:tl.constexpr,TAKE:tl.constexpr):
 win=tl.program_id(0);h=tl.program_id(1);q=tl.arange(0,W);w=tl.arange(0,P)
 bits=tl.load(BITS+(h*N+win*W+q[:,None])*WORDS+w[None,:],(win*W+q[:,None]<N)&(w[None,:]<WORDS),0).to(tl.uint32)
 remain=win*W+q<N
 for group in range(W//32-1):
  seed=tl.min(tl.where(remain,q,2147483647),0)
  united=tl.reduce(tl.where(q[:,None]==seed,bits,0),0,_or)
  distance=tl.sum(_popcount(bits^united[None,:]),1)
  for step in range(32//TAKE):
   added=tl.sum(_popcount(bits&~united[None,:]),1)
   key=(added.to(tl.uint32)<<20)|(distance.to(tl.uint32)<<10)|q.to(tl.uint32)
   key=tl.where(remain,key,0xffffffff);ordered=tl.sort(key,descending=False)
   threshold=tl.sum(tl.where(q==TAKE-1,ordered,0),0)
   chosen=remain&(key<=threshold)
   united=united|tl.reduce(tl.where(chosen[:,None],bits,0),0,_or)
   remain=remain&~chosen
   position=win*W+group*32+step*TAKE+q
   tl.store(OUT+h*N+position,win*W+(ordered&(W-1)).to(tl.int32),(q<TAKE)&(position<N))
 rank=tl.cumsum(remain.to(tl.int32),0)-1
 pos=win*W+W-32+rank
 tl.store(OUT+h*N+pos,win*W+q,remain&(pos<N))

def bitset_order(mask,method):
 h,n,b=mask.shape;parts=method.split('_');w=int(parts[1]);take=int(parts[2])if len(parts)>2 else 8;warps=int(parts[3])if len(parts)>3 else 1;words=tr.cdiv(b,32)
 bits=torch.empty((h,n,words),device=mask.device,dtype=torch.int32);out=torch.empty((h,n),device=mask.device,dtype=torch.int64)
 _pack[(tr.cdiv(n,32),h)](mask,bits,n,b,words,tr.next_power_of_2(b),num_warps=4)
 _group[(tr.cdiv(n,w),h)](bits,out,n,words,tr.next_power_of_2(words),w,take,num_warps=warps)
 return out
