"""Query aggregation and reduced fine routing; original-token assignment storage."""
import math,torch,triton as tr
import triton.language as tl
from ..opt26.route import coarse
from ..opt26.radix import top as radix_top
from ..opt17.route import _top
from ..opt28.retile_v2 import _merge

def top(score,count):
 h,t,b=score.shape
 if b>512:return radix_top(score,count,8,4)
 out=torch.empty((h,t,count),device=score.device,dtype=torch.int32)
 _top[(h*t,)](score,out,h*t,b,count,tr.next_power_of_2(b),num_warps=4)
 return out
@tr.jit
def _assign(Q,IX,W,BIAS,A,N:tl.constexpr,R:tl.constexpr,RR:tl.constexpr,IDENTITY:tl.constexpr):
 h=tl.program_id(1);i=tl.program_id(0)*32+tl.arange(0,32);d=tl.arange(0,64);r=tl.arange(0,RR)
 q=tl.load(Q+(h*N+i[:,None])*64+d[None,:],i[:,None]<N,0)
 w=tl.load(W+(h*R+r[None,:])*64+d[:,None],r[None,:]<R,0).to(tl.float16)
 bias=tl.load(BIAS+h*R+r,r<R,0)
 logits=tl.where(r[None,:]<R,tl.dot(q,w)+bias[None,:],float('-inf'));e=tl.exp(logits-tl.max(logits,1)[:,None]);a=e/tl.sum(e,1)[:,None]
 if IDENTITY:ix=i
 else:ix=tl.load(IX+h*N+i,i<N,0)
 tl.store(A+(h*N+ix[:,None])*R+r[None,:],a,(i[:,None]<N)&(r[None,:]<R))
def assign(q,ix,w,bias,out=None):
 h,n,_=q.shape;r=w.shape[1]
 if out is None:out=torch.empty((h,n,r),device=q.device,dtype=torch.float16)
 _assign[(tr.cdiv(n,32),h)](q,ix if ix is not None else q,w,bias,out,n,r,max(16,tr.next_power_of_2(r)),ix is None,num_warps=4)
 return out
@tr.jit
def _aggregate(Q,IX,A,O,M,N:tl.constexpr,T:tl.constexpr,G:tl.constexpr,R:tl.constexpr,RR:tl.constexpr,MODE:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);i=t*G+tl.arange(0,G);d=tl.arange(0,64);r=tl.arange(0,RR)
 q=tl.load(Q+(h*N+i[:,None])*64+d[None,:],i[:,None]<N,0)
 if MODE==0:
  ix=tl.load(IX+h*N+i,i<N,0);a=tl.load(A+(h*N+ix[:,None])*R+r[None,:],(i[:,None]<N)&(r[None,:]<R),0).to(tl.float32)
 elif MODE==1:
  a=((tl.arange(0,G)[:,None]//(G//R))==r[None,:]).to(tl.float32)*(i[:,None]<N)
 elif MODE==2:
  pos=tl.minimum(t*G+r*(G//R)+(G//R)//2,N-1)
  qq=tl.load(Q+(h*N+pos[:,None])*64+d[None,:],r[:,None]<R,0)
  tl.store(O+((h*T+t)*R+r[:,None])*64+d[None,:],qq,r[:,None]<R);tl.store(M+(h*T+t)*R+r,1./R,r<R)
 else:
  qq=tl.load(Q+(h*N+(t*G+r)[:,None])*64+d[None,:],((t*G+r)[:,None]<N)&(r[:,None]<R),0)
  mass=tl.where((t*G+r<N)&(r<R),1./tl.minimum(G,N-t*G),0.)
  tl.store(O+((h*T+t)*R+r[:,None])*64+d[None,:],qq,r[:,None]<R);tl.store(M+(h*T+t)*R+r,mass,r<R)
 if MODE<=1:
  count=tl.sum(a,0);value=tl.dot(tl.trans(a).to(tl.float16),q)/tl.maximum(count,1e-20)[:,None]
  tl.store(O+((h*T+t)*R+r[:,None])*64+d[None,:],value,r[:,None]<R)
  tl.store(M+(h*T+t)*R+r,count/tl.minimum(G,N-t*G),r<R)
def aggregate(q,ix,a,g,r,mode):
 h,n,_=q.shape;t=tr.cdiv(n,g);bar=torch.empty((h,t,r,64),device=q.device,dtype=torch.float16);mass=torch.empty((h,t,r),device=q.device)
 _aggregate[(t,h)](q,ix,a if a is not None else q,bar,mass,n,t,g,r,max(16,tr.next_power_of_2(r)),{'learned':0,'mean':1,'sample':2,'full':3}[mode],num_warps=4)
 return bar,mass
@tr.jit
def _part(Q,K,LC,IDS,S,T:tl.constexpr,R:tl.constexpr,RR:tl.constexpr,B:tl.constexpr,C:tl.constexpr,GP:tl.constexpr,P:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(2);ci=tl.program_id(1)*GP+tl.arange(0,GP);pr=tl.arange(0,GP*P);d=tl.arange(0,64);r=tl.arange(0,RR)
 ix=tl.program_id(1)*GP+pr//P;page=tl.load(IDS+(h*T+g)*C+ix,ix<C,0)
 q=tl.load(Q+((h*T+g)*R+r[:,None])*64+d[None,:],r[:,None]<R,0)
 k=tl.load(K+((h*B+page[None,:])*16+pr[None,:]%P)*64+d[:,None])
 ll=tl.dot(q,k).reshape((RR,GP,P))*.08838834764831845
 pc=tl.load(IDS+(h*T+g)*C+ci,ci<C,0);lc=tl.load(LC+(h*B+pc[:,None])*16+tl.arange(0,P)[None,:])
 ll+=lc[None,:,:];mx=tl.max(ll,2);score=mx+tl.log(tl.sum(tl.exp(ll-mx[:,:,None]),2))
 tl.store(S+((h*T+g)*R+r[:,None])*C+ci[None,:],score,(r[:,None]<R)&(ci[None,:]<C))
@tr.jit
def _fine(Q,K,LC,IDS,S,T:tl.constexpr,R:tl.constexpr,RR:tl.constexpr,B:tl.constexpr,C:tl.constexpr,PAGES:tl.constexpr,GP:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(2);ci=tl.program_id(1)*GP+tl.arange(0,GP);pc=tl.load(IDS+(h*T+g)*C+ci,ci<C,0)
 age=tr.cdiv(B,PAGES)-1-pc//PAGES;np=tl.where(age==0,16,tl.where(age==1,8,tl.where(age==2,4,2)));mp=tl.max(tl.where(ci<C,np,0),0)
 if mp<=2:_part(Q,K,LC,IDS,S,T,R,RR,B,C,GP,2)
 elif mp<=4:_part(Q,K,LC,IDS,S,T,R,RR,B,C,GP,4)
 elif mp<=8:_part(Q,K,LC,IDS,S,T,R,RR,B,C,GP,8)
 else:_part(Q,K,LC,IDS,S,T,R,RR,B,C,GP,16)
@tr.jit
def _mass(F,W,IDS,O,T:tl.constexpr,R:tl.constexpr,B:tl.constexpr,C:tl.constexpr,RR:tl.constexpr,CC:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);r=tl.arange(0,RR);c=tl.arange(0,CC)
 x=tl.load(F+((h*T+t)*R+r[:,None])*C+c[None,:],(r[:,None]<R)&(c[None,:]<C),float('-inf'));mx=tl.max(x,1);mx=tl.where(r<R,mx,0.);e=tl.exp(x-mx[:,None]);prob=e/tl.maximum(tl.sum(e,1),1e-20)[:,None]
 w=tl.load(W+(h*T+t)*R+r,r<R,0);value=tl.sum(prob*w[:,None],0);page=tl.load(IDS+(h*T+t)*C+c,c<C,0)
 tl.store(O+(h*T+t)*B+page,value,c<C)
def route(q,state,bar,mass,seg,g,k,fraction):
 h,n,_=q.shape;t=bar.shape[1];r=bar.shape[2];b64=state['key'].shape[1];c=math.ceil(b64*fraction)
 cs=coarse(q,state,8,128 if b64>1024 else 64,4)
 if g==64:
  cs=torch.nn.functional.pad(cs,(0,0,0,cs.shape[1]%2),value=-torch.inf).reshape(h,t,2,b64).amax(2)
 ids=top(cs,c);fine=torch.empty((h,t,r,c),device=q.device)
 _fine[(t,tr.cdiv(c,8),h)](bar,state['key'],state['logcount'],ids,fine,t,r,max(16,tr.next_power_of_2(r)),b64,c,tr.cdiv(seg,64),8,num_warps=4,num_stages=1)
 imp=torch.zeros((h,t,b64),device=q.device)
 _mass[(t,h)](fine,mass,ids,imp,t,r,b64,c,tr.next_power_of_2(r),tr.next_power_of_2(c),num_warps=4)
 if k==128:
  p64=tr.cdiv(seg,64);pk=tr.cdiv(seg,k);b=b64//p64*pk;score=torch.empty((h,t,b),device=q.device)
  _merge[(t,h)](imp,score,b64,b,t,t,1,p64,pk,k,tr.next_power_of_2(b),num_warps=4)
 else:score=imp;b=b64
 chosen=top(score,math.ceil(b*.2))
 return chosen,dict(fine=fine,candidates=ids,coarse=cs,importance=score,logical_importance=imp,bar=bar,mass=mass)
