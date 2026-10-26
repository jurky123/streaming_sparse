"""Parallelize H/C and remove/add candidate pools; retain packed32 ordering."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _select(X,S,O,REM,ADD,N:tl.constexpr,B:tl.constexpr,P:tl.constexpr,BETA:tl.constexpr):
 r=tl.program_id(0);p=tl.arange(0,P);pool=tl.arange(0,4)
 x=tl.load(X+r*B+p,p<B,0.)
 mean=tl.sum(x,0)/B
 std=tl.sqrt(tl.sum(tl.where(p<B,(x-mean)*(x-mean),0.),0)/B)
 selected=(p<B)&((x>mean+BETA*std)|(std<=1.e-6))
 if tl.sum(selected.to(tl.int32),0)==0:
  maximum=tl.max(tl.where(p<B,x,float('-inf')),0)
  best=tl.min(tl.where((p<B)&(x==maximum),p,2147483647),0)
  selected=p==best
 tl.store(S+r*B+p,selected,p<B)
 origin=tl.load(O+p,p<B,-1)
 distance=tl.abs(x-mean-BETA*std)
 key=(distance.to(tl.uint32,bitcast=True)&(0xffffffff^(P-1)))|p.to(tl.uint32)
 remain=(p[None,:]<B)&tl.where(pool[:,None]<2,selected[None,:],~selected[None,:])&(origin[None,:]==pool[:,None]%2)
 result=tl.full((4,4),-1,tl.int32);columns=tl.arange(0,4)
 for c in tl.static_range(3):
  winning=tl.min(tl.where(remain,key[None,:],0x7fffffff),1)
  valid=winning!=0x7fffffff;index=(winning&(P-1)).to(tl.int32)
  result=tl.where(columns[None,:]==c,tl.where(valid,index,-1)[:,None],result)
  remain=remain&~(valid[:,None]&(p[None,:]==index[:,None]))
 side=tl.arange(0,2);col=tl.arange(0,8)
 # Recombine the disjoint H/C pools for each side's two exploration candidates.
 remleft=tl.sum(tl.where(pool[:,None]<2,remain,False).to(tl.int32),0)>0
 addleft=tl.sum(tl.where(pool[:,None]>=2,remain,False).to(tl.int32),0)>0
 left=tl.where(side[:,None]==0,remleft[None,:],addleft[None,:])
 out=tl.full((2,8),-1,tl.int32)
 for c in tl.static_range(6):
  picked=tl.sum(tl.where(columns[None,:]==c%3,result,0),1)
  rr=tl.sum(tl.where(pool==c//3,picked,0),0)
  aa=tl.sum(tl.where(pool==2+c//3,picked,0),0)
  value=tl.where(side==0,rr,aa)
  out=tl.where(col[None,:]==c,value[:,None],out)
 q=r%N;h=r//N;tile_n=tl.minimum(128,N-q//128*128);canonical=h*tile_n+q%128
 raw=p.to(tl.uint64)*1103515245+canonical.to(tl.uint64)*12345+16
 folded=(raw&2147483647)+(raw>>31)
 hashes=tl.where(folded>=2147483647,folded-2147483647,folded).to(tl.float32)
 hkey=(hashes.to(tl.uint32,bitcast=True)&(0xffffffff^(P-1)))|p.to(tl.uint32)
 for c in tl.static_range(6,8):
  winning=tl.min(tl.where(left,hkey[None,:],0x7fffffff),1)
  valid=winning!=0x7fffffff;index=(winning&(P-1)).to(tl.int32)
  out=tl.where(col[None,:]==c,tl.where(valid,index,-1)[:,None],out)
  left=left&~(valid[:,None]&(p[None,:]==index[:,None]))
 if tl.min(tl.min(out,1),0)<0:
  for c in tl.static_range(8):
   missing=tl.sum(tl.where(col[None,:]==c,out,0),1)<0
   winning=tl.min(tl.where(left,key[None,:],0x7fffffff),1)
   valid=missing&(winning!=0x7fffffff);index=(winning&(P-1)).to(tl.int32)
   out=tl.where((col[None,:]==c)&valid[:,None],index[:,None],out)
   left=left&~(valid[:,None]&(p[None,:]==index[:,None]))
 rr=tl.sum(tl.where(side[:,None]==0,out,0),0);aa=tl.sum(tl.where(side[:,None]==1,out,0),0)
 tl.store(REM+r*8+col,rr);tl.store(ADD+r*8+col,aa)

def select(score,origins,beta,warps=4):
 h,n,b=score.shape;mask=torch.empty_like(score,dtype=torch.bool);rem=torch.empty((h*n,8),device=score.device,dtype=torch.int64);add=torch.empty_like(rem)
 _select[(h*n,)](score,mask,origins,rem,add,n,b,tr.next_power_of_2(b),beta,num_warps=warps,enable_fp_fusion=False)
 return mask,rem,add
