"""Multiple independent query rows per CTA; no mask or score sharing."""
import torch,triton as tr
import triton.language as tl
from .select import _merge3,_merge2
@tr.jit
def _pick(rank,allowed,p,P:tl.constexpr):
 key=(rank.to(tl.uint32,bitcast=True)&(0xffffffff^(P-1)))|p[None,:].to(tl.uint32)
 best=tl.min(tl.where(allowed,key,0x7fffffff),1)
 return tl.where(best!=0x7fffffff,(best&(P-1)).to(tl.int32),2147483647)
@tr.jit
def _select(X,S,O,REM,ADD,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,BETA:tl.constexpr,ROWS:tl.constexpr,G:tl.constexpr):
 r=tl.program_id(0)*G+tl.arange(0,G);p=tl.arange(0,P)
 x=tl.load(X+r[:,None]*B+p[None,:],(r[:,None]<ROWS)&(p[None,:]<B),0.)
 mean=tl.sum(x,1)/B;std=tl.sqrt(tl.sum(tl.where(p[None,:]<B,(x-mean[:,None])*(x-mean[:,None]),0.),1)/B)
 selected=(p[None,:]<B)&((x>mean[:,None]+BETA*std[:,None])|(std[:,None]<=1.e-6))
 empty=tl.sum(selected.to(tl.int32),1)==0
 if tl.sum(empty.to(tl.int32),0)>0:
  maximum=tl.max(tl.where(p[None,:]<B,x,float('-inf')),1)
  best=tl.min(tl.where((p[None,:]<B)&(x==maximum[:,None]),p[None,:],2147483647),1)
  selected=tl.where(empty[:,None],p[None,:]==best[:,None],selected)
 tl.store(S+r[:,None]*B+p[None,:],selected,(r[:,None]<ROWS)&(p[None,:]<B))
 origin=tl.load(O+p,p<B,-1);distance=tl.abs(x-mean[:,None]-BETA*std[:,None])
 q=r%N;h=r//N;tile_n=tl.minimum(128,N-q//128*128);canonical=h*tile_n+q%128
 raw=p[None,:].to(tl.uint64)*1103515245+canonical[:,None].to(tl.uint64)*12345+16;folded=(raw&2147483647)+(raw>>31)
 hashes=tl.where(folded>=2147483647,folded-2147483647,folded).to(tl.float32)
 cols=tl.arange(0,8)
 for side in tl.static_range(2):
  remain=(p[None,:]<B)&(selected if side==0 else ~selected);result=tl.full((G,8),-1,tl.int32)
  keys=(distance.to(tl.uint32,bitcast=True)&(0xffffffff^(P-1)))|p[None,:].to(tl.uint32)
  hashkeys=(hashes.to(tl.uint32,bitcast=True)&(0xffffffff^(P-1)))|p[None,:].to(tl.uint32)
  sentinel=tl.full((G,P),0x7fffffff,tl.uint32)
  for pool in tl.static_range(2):
   ranked=tl.where(remain&(origin[None,:]==pool),keys,sentinel)
   k0,k1,k2=tl.reduce((ranked,sentinel,sentinel),1,_merge3)
   for c in tl.static_range(3):
    key=k0 if c==0 else(k1 if c==1 else k2);idx=(key&(P-1)).to(tl.int32);valid=key!=0x7fffffff
    result=tl.where(cols[None,:]==pool*3+c,tl.where(valid[:,None],idx[:,None],-1),result)
    remain=remain&~(valid[:,None]&(p[None,:]==idx[:,None]))
  h0,h1=tl.reduce((tl.where(remain,hashkeys,sentinel),sentinel),1,_merge2)
  for c in tl.static_range(2):
   key=h0 if c==0 else h1;idx=(key&(P-1)).to(tl.int32);valid=key!=0x7fffffff
   result=tl.where(cols[None,:]==6+c,tl.where(valid[:,None],idx[:,None],-1),result)
   remain=remain&~(valid[:,None]&(p[None,:]==idx[:,None]))
  if tl.min(tl.min(result,1),0)<0:
   for c in tl.static_range(8):
    missing=tl.sum(tl.where(cols[None,:]==c,result,0),1)<0
    idx=_pick(distance,remain,p,P);valid=missing&(idx<2147483647)
    result=tl.where((cols[None,:]==c)&valid[:,None],idx[:,None],result)
    remain=remain&~(valid[:,None]&(p[None,:]==idx[:,None]))
  if side==0:tl.store(REM+r[:,None]*8+cols[None,:],result,r[:,None]<ROWS)
  else:tl.store(ADD+r[:,None]*8+cols[None,:],result,r[:,None]<ROWS)

def select(score,origins,beta,group=4,warps=4):
 h,n,b=score.shape;mask=torch.empty_like(score,dtype=torch.bool);rem=torch.empty((h*n,8),device=score.device,dtype=torch.int64);add=torch.empty_like(rem)
 _select[(tr.cdiv(h*n,group),)](score,mask,origins,rem,add,n,b,tr.next_power_of_2(b),beta,h*n,group,num_warps=warps,enable_fp_fusion=False)
 return mask,rem,add
