import numpy as np
class ResidualProblem:
 def __init__(self,root):
  self.root=root;r=dict(np.load(root/'residuals_fp64.npz'));self.e=r['e'];self.d=r['d'];self.od=r['dense_output'];self.C=float(np.square(self.od).sum()+1e-12);self.Q,self.P,self.D=self.e.shape;self.K=int(r['K']);self.own=int(r['own']);self.free=np.delete(np.arange(self.P),self.own);self.data=dict(mass=r['mass'],risk=r['risk'],Z=r['exact_mass'],Zh=r['summary_mass']);self.G=self.e@self.e.transpose(0,2,1);self.diag=np.diagonal(self.G,axis1=1,axis2=2)
  assert (self.Q,self.D)==(64,128) and self.K==int(self.P*.1) and np.isfinite(self.e).all()
 def state(self,x):
  E=self.e[:,~x].sum(1);z=np.where(x[None],self.data['Z'],self.data['Zh']).sum(1);return E,z,float(np.sum(np.square(E).sum(1)/z**2)/self.C)
 def value(self,x):return self.state(x)[2]
 def swaps(self,x,direct=False):
  r=np.where(x & (np.arange(self.P)!=self.own))[0];a=np.where(~x)[0];E,z,_=self.state(x)
  if direct:
   rz=r.repeat(len(a));az=np.tile(a,len(r));vals=[]
   for i in range(0,len(rz),128):
    rr,aa=rz[i:i+128],az[i:i+128];v=E[:,None,:]+self.e[:,rr]-self.e[:,aa];zz=z[:,None]-self.d[:,rr]+self.d[:,aa];assert (zz>0).all();vals.append(np.sum(np.square(v).sum(-1)/zz**2,axis=0)/self.C)
   return r,a,np.concatenate(vals).reshape(len(r),len(a))
  h=np.einsum('qd,qpd->qp',E,self.e);n=np.square(E).sum(-1)[:,None,None]+2*(h[:,r,None]-h[:,None,a])+self.diag[:,r,None]+self.diag[:,None,a]-2*self.G[:,r[:,None],a[None,:]];zz=z[:,None,None]-self.d[:,r,None]+self.d[:,None,a];assert (zz>0).all();return r,a,np.sum(n/zz**2,axis=0)/self.C
 def bounds(self,fixed=None):
  fixed={self.own:0} if fixed is None else dict(fixed)|{self.own:0};free=np.array([p for p in range(self.P) if p not in fixed]);one=[p for p,v in fixed.items() if v==1];m=self.P-self.K-len(one);assert 0<=m<=len(free);vals=[]
  for largest in (True,False):
   y=np.zeros((self.Q,self.P),bool);y[:,one]=True;order=np.argsort(self.d[:,free],axis=1)
   if largest:order=order[:,::-1]
   if m:y[np.arange(self.Q)[:,None],free[order[:,:m]]]=True
   vals.append(np.where(y,self.data['Zh'],self.data['Z']).sum(1))
  L,U=vals;assert (L>0).all() and (U>=L).all();return np.nextafter(L*(1-1e-13),0),np.nextafter(U*(1+1e-13),np.inf)
