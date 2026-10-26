"""Associative top-3/top-2 reduction: remove serial candidate dependencies."""
import torch
import triton as tr
import triton.language as tl
from ..opt9.select import _pick

@tr.jit
def _merge3(a0,a1,a2,b0,b1,b2):
 lo=tl.minimum(a1,b1);hi=tl.maximum(a0,b0)
 return tl.minimum(a0,b0),tl.minimum(hi,lo),tl.minimum(tl.minimum(tl.maximum(hi,lo),tl.maximum(a1,b1)),tl.minimum(a2,b2))

@tr.jit
def _merge2(a0,a1,b0,b1):
 return tl.minimum(a0,b0),tl.minimum(tl.maximum(a0,b0),tl.minimum(a1,b1))

@tr.jit
def _select(X,S,O,REM,ADD,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,BETA:tl.constexpr):
 r=tl.program_id(0);p=tl.arange(0,P);cols=tl.arange(0,8)
 x=tl.load(X+r*B+p,p<B,0.);mean=tl.sum(x,0)/B
 std=tl.sqrt(tl.sum(tl.where(p<B,(x-mean)*(x-mean),0.),0)/B)
 selected=(p<B)&((x>mean+BETA*std)|(std<=1.e-6))
 if tl.sum(selected.to(tl.int32),0)==0:
  maximum=tl.max(tl.where(p<B,x,float('-inf')),0)
  best=tl.min(tl.where((p<B)&(x==maximum),p,2147483647),0);selected=p==best
 tl.store(S+r*B+p,selected,p<B)
 origin=tl.load(O+p,p<B,-1);distance=tl.abs(x-mean-BETA*std)
 keys=(distance.to(tl.uint32,bitcast=True)&(0xffffffff^(P-1)))|p.to(tl.uint32)
 q=r%N;h=r//N;tile_n=tl.minimum(128,N-q//128*128);canonical=h*tile_n+q%128
 raw=p.to(tl.uint64)*1103515245+canonical.to(tl.uint64)*12345+16
 folded=(raw&2147483647)+(raw>>31);hashes=tl.where(folded>=2147483647,folded-2147483647,folded).to(tl.float32)
 hashkeys=(hashes.to(tl.uint32,bitcast=True)&(0xffffffff^(P-1)))|p.to(tl.uint32)
 sentinel=tl.full((P,),0x7fffffff,tl.uint32)
 for side in tl.static_range(2):
  remain=(p<B)&(selected if side==0 else ~selected);result=tl.full((8,),-1,tl.int32)
  for pool in tl.static_range(2):
   ranked=tl.where(remain&(origin==pool),keys,sentinel)
   k0,k1,k2=tl.reduce((ranked,sentinel,sentinel),0,_merge3)
   for c in tl.static_range(3):
    key=k0 if c==0 else (k1 if c==1 else k2)
    idx=(key&(P-1)).to(tl.int32);valid=key!=0x7fffffff
    result=tl.where(cols==pool*3+c,tl.where(valid,idx,-1),result);remain=remain&~(valid&(p==idx))
  h0,h1=tl.reduce((tl.where(remain,hashkeys,sentinel),sentinel),0,_merge2)
  for c in tl.static_range(2):
   key=h0 if c==0 else h1;idx=(key&(P-1)).to(tl.int32);valid=key!=0x7fffffff
   result=tl.where(cols==6+c,tl.where(valid,idx,-1),result);remain=remain&~(valid&(p==idx))
  if tl.min(result,0)<0:
   for c in tl.static_range(8):
    missing=tl.sum(tl.where(cols==c,result,0),0)<0
    idx=_pick(distance,remain,p,P,'packed32');valid=missing&(idx<2147483647)
    result=tl.where((cols==c)&valid,idx,result);remain=remain&~(valid&(p==idx))
  if side==0:tl.store(REM+r*8+cols,result)
  else:tl.store(ADD+r*8+cols,result)

def select(score,origins,beta,warps=4):
 h,n,b=score.shape;mask=torch.empty_like(score,dtype=torch.bool);rem=torch.empty((h*n,8),device=score.device,dtype=torch.int64);add=torch.empty_like(rem)
 _select[(h*n,)](score,mask,origins,rem,add,n,b,tr.next_power_of_2(b),beta,num_warps=warps,enable_fp_fusion=False)
 return mask,rem,add
