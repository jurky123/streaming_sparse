import numpy as np,itertools,time
from ..e11_oracle.solver import objective
class Problem:
 def __init__(self,data,K=41,require_full=True):
  self.data=data;self.Q,self.P,self.D=data['N'].shape;self.own=int(data['own']);self.K=K
  if require_full:assert (self.Q,self.P,self.D,self.K)==(64,410,128,41)
  self.zd=data['Z'].sum(1);self.od=data['N'].sum(1)/self.zd[:,None];self.C=float(np.square(self.od).sum()+1e-12)
  self.d=(data['Z']-data['Zh'])/self.zd[:,None];self.e=(data['N']-data['Nh'])/self.zd[:,None,None]-self.od[:,None,:]*self.d[:,:,None]
  self.G=self.e@self.e.transpose(0,2,1);self.diag=np.diagonal(self.G,axis1=1,axis2=2);self.free=np.delete(np.arange(self.P),self.own)
 def value(self,x):return objective(self.data,x)
 def state(self,x):
  E=self.e[:,~x].sum(1);z=np.where(x[None],self.data['Z'],self.data['Zh']).sum(1)/self.zd
  return E,z,float(np.sum(np.square(E).sum(1)/z**2)/self.C)
 def gram_value(self,x):
  y=(~x).astype(float);n=np.einsum('p,qpr,r->q',y,self.G,y,optimize=True);z=np.where(x[None],self.data['Z'],self.data['Zh']).sum(1)/self.zd
  return float(np.sum(n/z**2)/self.C)
 def swaps(self,x,direct=False):
  r=np.where(x & (np.arange(self.P)!=self.own))[0];a=np.where(~x)[0];assert len(r)*len(a)==14760
  E,z,_=self.state(x);rz=r.repeat(len(a));az=np.tile(a,len(r))
  if direct:
   vals=[]
   for i in range(0,len(rz),128):
    rr,aa=rz[i:i+128],az[i:i+128];v=E[:,None,:]+self.e[:,rr]-self.e[:,aa];zz=z[:,None]-self.d[:,rr]+self.d[:,aa];assert (zz>0).all();vals.append(np.sum(np.square(v).sum(-1)/zz**2,axis=0)/self.C)
   return r,a,np.concatenate(vals).reshape(len(r),len(a))
  h=np.einsum('qd,qpd->qp',E,self.e);n=np.square(E).sum(-1)[:,None,None]+2*(h[:,r,None]-h[:,None,a])+self.diag[:,r,None]+self.diag[:,None,a]-2*self.G[:,r[:,None],a[None,:]]
  zz=z[:,None,None]-self.d[:,r,None]+self.d[:,None,a];assert (zz>0).all();return r,a,np.sum(n/zz**2,axis=0)/self.C
 def descend(self,start,label,max_steps=1000):
  begin=time.monotonic();x=start.copy();trace=[];initial=self.value(x);cert=None
  for step in range(max_steps):
   v=self.value(x);tol=max(1e-12,1e-9*abs(v));r,a,scores=self.swaps(x);idx=np.unravel_index(np.argmin(scores),scores.shape)
   if scores[idx]>=v-tol:
    _,_,ds=self.swaps(x,direct=True);err=float(np.max(np.abs(ds-scores)));assert err<max(1e-11,1e-8*abs(v)),err;idx=np.unravel_index(np.argmin(ds),ds.shape)
    if ds[idx]>=v-tol:
     cert=dict(status='ONE_SWAP_LOCAL_TOL',neighbors=14760,best_neighbor=float(ds[idx]),objective=v,tolerance=tol,max_gram_direct_swap_error=err);break
   xx=x.copy();xx[r[idx[0]]]=False;xx[a[idx[1]]]=True;vv=self.value(xx);res=self.state(xx)[2]
   assert abs(vv-res)<max(1e-12,1e-9*abs(vv));assert vv<v-tol,(vv,v)
   trace.append(dict(step=step,remove=int(r[idx[0]]),add=int(a[idx[1]]),before=v,after=vv));x=xx
  return x,dict(label=label,initial=initial,final=self.value(x),steps=len(trace),trace=trace,local_certificate=cert,wall_seconds=time.monotonic()-begin,mask=np.where(x)[0].tolist())
 def two_swap(self,x):
  r,a,s=self.swaps(x);rr=r[np.argsort(s.min(1))[:10]];aa=a[np.argsort(s.min(0))[:20]]
  rp=np.array(list(itertools.combinations(rr,2)));ap=np.array(list(itertools.combinations(aa,2)));changes=np.c_[np.repeat(rp,len(ap),axis=0),np.tile(ap,(len(rp),1))];sgn=np.array([1,1,-1,-1]);E,z,v=self.state(x);h=np.einsum('qd,qpd->qp',E,self.e);n=np.square(E).sum(-1)[:,None]+2*np.sum(h[:,changes]*sgn,axis=-1)
  for i in range(4):
   for j in range(4):n+=sgn[i]*sgn[j]*self.G[:,changes[:,i],changes[:,j]]
  zz=z[:,None]-np.sum(self.d[:,changes]*sgn,axis=-1);vals=np.sum(n/zz**2,axis=0)/self.C;ix=int(np.argmin(vals));xx=x.copy();xx[changes[ix,:2]]=False;xx[changes[ix,2:]]=True;actual=self.value(xx)
  if actual>=v-max(1e-12,1e-9*v):xx=x.copy();actual=v
  return xx,dict(pool_selected=rr.tolist(),pool_unselected=aa.tolist(),candidates=len(changes),before=v,after=actual,global_or_two_swap_certificate=False)
 def bounds(self,fixed=None):
  # y=1 is summarized; own y=0. Positive sums avoid subtraction cancellation.
  fixed={self.own:0} if fixed is None else dict(fixed)|{self.own:0};free=np.array([p for p in range(self.P) if p not in fixed]);one=[p for p,v in fixed.items() if v==1];m=self.P-self.K-len(one);assert 0<=m<=len(free)
  vals=[]
  for largest in (True,False):
   y=np.zeros((self.Q,self.P),bool);y[:,one]=True;order=np.argsort(self.d[:,free],axis=1)
   if largest:order=order[:,::-1]
   if m:y[np.arange(self.Q)[:,None],free[order[:,:m]]]=True
   vals.append(np.where(y,self.data['Zh'],self.data['Z']).sum(1)/self.zd)
  L,U=vals;assert (L>0).all() and (U>=L).all()
  # Outward safety margin only weakens the lower-bound relaxation.
  U=np.nextafter(U*(1+1e-13),np.inf);L=np.nextafter(L*(1-1e-13),0)
  return L,U
 def decomposition(self,x):
  E,z,v=self.state(x);A=np.square(self.e[:,~x]).sum((1,2));cross=np.square(E).sum(1)-A
  return dict(A=A,C=cross,kappa=np.square(E).sum(1)/(A+1e-12),denominator_ratio=z,Jq=np.square(E).sum(1)/z**2,A_nmse=float(np.sum(A/z**2)/self.C),C_nmse=float(np.sum(cross/z**2)/self.C),fixed_dense_denominator_nmse=float(np.square(E).sum()/self.C),nmse=v)
