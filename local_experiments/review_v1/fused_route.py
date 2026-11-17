"""Three-kernel implementation of latest RouteOps.features + theta/probe mean.
Same four risk terms and BF16 deployed operands; FP32 accumulation/reductions.
No legacy gain, proxy route, threshold, or weight substitution.
"""
import math,torch
import triton as tr
import triton.language as tl

@tr.jit
def _project(Q,K,Z,N:tl.constexpr,P:tl.constexpr,D:tl.constexpr,BQ:tl.constexpr=32,BR:tl.constexpr=64):
 h=tl.program_id(2);qi=tl.program_id(0)*BQ+tl.arange(0,BQ);ri=tl.program_id(1)*BR+tl.arange(0,BR);d=tl.arange(0,D)
 q=tl.load(Q+(h*N+qi[:,None])*D+d[None,:],qi[:,None]<N,0)
 k=tl.load(K+(h*P*4+ri[None,:])*D+d[:,None],ri[None,:]<P*4,0)
 z=tl.dot(q,k)*(1./math.sqrt(D))
 tl.store(Z+(h*N+qi[:,None])*(P*4)+ri[None,:],z,(qi[:,None]<N)&(ri[None,:]<P*4))

@tr.jit
def _normalizer(Q,Z,ST,LZ,MOM,BMAX,N:tl.constexpr,P:tl.constexpr,D:tl.constexpr,R:tl.constexpr,PP:tl.constexpr):
 i=tl.program_id(0);h=tl.program_id(1);r=tl.arange(0,R)
 z=tl.load(Z+(h*N+i)*(P*4)+r,r<P*4,-float('inf'));mx=tl.max(z,0);lz=mx+tl.log(tl.sum(tl.exp(z-mx),0));tl.store(LZ+h*N+i,lz)
 d=tl.arange(0,D);q=tl.load(Q+(h*N+i)*D+d).to(tl.float32);sq=q*q;groups=tl.reshape(sq,(8,16));q2=tl.sum(groups,1)/D;g=tl.arange(0,8)
 tl.store(MOM+(h*N+i)*9+g,q2);tl.store(MOM+(h*N+i)*9+8,tl.sqrt(tl.sum(sq,0))/(D**0.5))
 if i==0:
  p=tl.arange(0,PP);bp=tl.load(ST+((h*P+p)*4+1)*12+11,p<P,0).to(tl.float32);tl.store(BMAX+h,tl.max(bp,0))

@tr.jit
def _risk(Z,LZ,MOM,ST,THETA,BMAX,OUT,N:tl.constexpr,P:tl.constexpr,PROBES:tl.constexpr=16,BP:tl.constexpr=16):
 t=tl.program_id(0);h=tl.program_id(2);p=tl.program_id(1)*BP+tl.arange(0,BP);j=tl.arange(0,PROBES);r=tl.arange(0,4);qi=t*PROBES+j
 off=((h*N+qi[:,None,None])*(P*4)+p[None,:,None]*4+r[None,None,:])
 z=tl.load(Z+off,p[None,:,None]<P,-float('inf'));lz=tl.load(LZ+h*N+qi)
 prob=tl.exp(z-lz[:,None,None]);alpha=tl.sum(prob,2)
 so=((h*P+p[:,None])*4+r[None,:])*12
 u=tl.zeros((PROBES,BP,4),tl.float32)
 for g in tl.static_range(8):
  q2=tl.load(MOM+(h*N+qi)*9+g);variance=tl.load(ST+so+g,p[:,None]<P,0).to(tl.float32)
  u=u+q2[:,None,None]*variance[None,:,:]
 u=tl.maximum(u,0.)
 qn=tl.load(MOM+(h*N+qi)*9+8);bk=tl.load(ST+so+8,p[:,None]<P,0).to(tl.float32);bv=tl.load(ST+so+9,p[:,None]<P,0).to(tl.float32);sv=tl.load(ST+so+10,p[:,None]<P,0).to(tl.float32)
 beta=qn[:,None,None]*bk[None,:,:]
 bv0=tl.load(ST+((h*P+p)*4)*12+11,p<P,0).to(tl.float32);bp=tl.load(ST+((h*P+p)*4+1)*12+11,p<P,0).to(tl.float32);c=bp+tl.load(BMAX+h)
 x0=alpha*bv0[None,:]+tl.sum(tl.abs(prob-alpha[:,:,None]/4.)*bv[None,:,:],2)
 x1=tl.sum(prob*tl.sqrt(u)*sv[None,:,:],2)
 x2=c[None,:]*tl.sum(prob*beta,2)
 x3=c[None,:]*tl.sum(prob*(u+beta*beta),2)
 w0=tl.load(THETA+h*4);w1=tl.load(THETA+h*4+1);w2=tl.load(THETA+h*4+2);w3=tl.load(THETA+h*4+3)
 risk=((x0*w0+x1*w1)+x2*w2)+x3*w3;score=tl.sum(risk,0)/PROBES
 tl.store(OUT+(h*(N//PROBES)+t)*P+p,score,p<P)

class FusedRiskRoute:
 def __init__(self,probes=16):
  assert probes==16,'Validated configuration uses sixteen probes'
  self.probes=probes;self.calls=0;self.entries={}
 @torch.no_grad()
 def __call__(self,q,s,theta):
  h,n,d=q.shape;p=s.keys.shape[1];assert d==128 and n%self.probes==0
  q=q.contiguous();key=s.keys.contiguous();st=s.stats.contiguous();theta=theta.contiguous()
  assert q.dtype==key.dtype==torch.bfloat16 and st.dtype==torch.bfloat16
  z=torch.empty(h,n,p*4,device=q.device,dtype=torch.float32);lz=torch.empty(h,n,device=q.device);mom=torch.empty(h,n,9,device=q.device);bmax=torch.empty(h,device=q.device);out=torch.empty(h,n//self.probes,p,device=q.device)
  _project[(tr.cdiv(n,32),tr.cdiv(p*4,64),h)](q,key,z,n,p,d,num_warps=4,num_stages=2,enable_fp_fusion=False)
  _normalizer[(n,h)](q,z,st,lz,mom,bmax,n,p,d,tr.next_power_of_2(p*4),tr.next_power_of_2(p),num_warps=4,enable_fp_fusion=False)
  _risk[(n//self.probes,tr.cdiv(p,16),h)](z,lz,mom,st,theta,bmax,out,n,p,self.probes,num_warps=4,enable_fp_fusion=False)
  self.calls+=1;self.entries[(n,p)]=True
  return out
