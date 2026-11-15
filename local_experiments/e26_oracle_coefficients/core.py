"""Analytic E26-A only. No neural training and no test data."""
from ..e25_decision_diagnostics.run import *
E26=ROOT/'artifacts/e26_oracle_coefficients_v1'
RIDGE=1e-6

def features(reader,q,c):
 rel=c['logit']-c['logit'].mean(-1,keepdim=True)
 phi=torch.tanh(reader.query(q.float())[:,None]/math.sqrt(128)+rel).float().double()
 return torch.cat([torch.ones_like(phi[...,:1]),phi],-1)

def ridge_fit(X,Y,scale):
 # X[P,Q,5], Y[P,Q,D]. Fixed train scale; additional page max scaling
 # cancels algebraically and solely prevents overflow in the normal equations.
 Yn=Y/scale;ys=Yn.abs().amax((1,2)).clamp_min(1e-300);Yn=Yn/ys[:,None,None]
 G=X.transpose(1,2)@X;lam=RIDGE*G.diagonal(dim1=-2,dim2=-1).sum(-1)/5
 eye=torch.eye(5,dtype=X.dtype,device=X.device);Greg=G+lam[:,None,None]*eye
 coef=torch.linalg.solve(Greg,X.transpose(1,2)@Yn)
 residual=Greg@coef-X.transpose(1,2)@Yn
 norm=(X.transpose(1,2)@Yn).abs().amax((1,2)).clamp_min(1e-300)
 kkt=residual.abs().amax((1,2))/norm
 coef=coef*(ys*scale)[:,None,None];pred=X@coef
 vals=torch.linalg.eigvalsh(G);regcond=(vals[:,-1]+lam)/(vals[:,0]+lam)
 return coef,pred,dict(kkt_max=float(kkt.max()),regularized_condition_max=float(regcond.max()),unregularized_rank_mean=float((vals>vals[:,-1,None]*1e-10).sum(-1).double().mean()))

def fit_coeff(F,logmass,target,beta,zscale,bscale):
 # Original design is diag(mass)[1,phi], not simply [1,phi].
 # Remove the page-wise maximum logmass before fitting; ridge is relative to
 # Gram trace, so this is algebraically invariant to the column scale change.
 lm=logmass.max(0).values
 X=(logmass-lm[None]).exp()[...,None]*F
 Cz,zfit,az=ridge_fit(X.permute(1,0,2),target.permute(1,0,2),zscale)
 Cb,bfit,ab=ridge_fit(F.permute(1,0,2),beta.T[...,None],bscale)
 return zfit.permute(1,0,2),bfit[...,0].T,dict(coef_z_scaled=Cz,coef_beta=Cb,logmass_page_scale=lm),dict(**{'z_'+k:v for k,v in az.items()},**{'beta_'+k:v for k,v in ab.items()})

def svd_rank5(target):
 Y=target.permute(1,0,2);s=Y.abs().amax((1,2)).clamp_min(1e-300);Yn=Y/s[:,None,None];U,S,Vh=torch.linalg.svd(Yn,full_matrices=False);fit=((U[:,:,:5]*S[:,None,:5])@Vh[:,:5])*s[:,None,None]
 return fit.permute(1,0,2),S

def safe_predictions(z,beta,logmass,mask,r,a,normalizer):
 # Common per-query scale for numerators/denominators, direct positive sums.
 # Avoid exp(beta) overflow, 0*inf, and subtracting a dominant selected page.
 le=logmass+beta.double();shift=torch.maximum(logmass.max(1).values,le.max(1).values);shift=torch.maximum(shift,z.abs().amax((1,2)).clamp_min(1e-300).log())
 mass=(logmass-shift[:,None]).exp();exact=(le-shift[:,None]).exp();zz=z*torch.exp(-shift)[:,None,None]
 allD=exact.sum(1);target=zz.sum(1)/allD[:,None]
 def obj(masks):
  mf=masks.double();den=exact@mf.T+mass@(1-mf).T
  if not bool((den>0).all()):raise FloatingPointError('nonpositive model denominator')
  ts=torch.einsum('qpd,ap->qad',zz,mf)
  return (target[:,None]-ts/den[...,None]).square().sum(-1).mean(0)/normalizer
 base=obj(mask[None])[0];scores=[]
 for st in range(0,len(r),32):
  rr,aa=r[st:st+32],a[st:st+32];m=mask[None].expand(len(rr),-1).clone();idx=torch.arange(len(rr),device=z.device);m[idx,rr]=False;m[idx,aa]=True;scores.append(base-obj(m))
 score=torch.cat(scores)
 # Exactly the same information for the independent net-residual-energy rule.
 net=zz-target[:,None]*(exact-mass)[...,None]
 energy=(net/allD[:,None,None]).square().sum(-1).mean(0)
 ind=energy[a]-energy[r]
 if not bool(torch.isfinite(score).all() and torch.isfinite(ind).all()):raise FloatingPointError('nonfinite predicted score')
 return score,ind

def nmse(pred,target):
 scale=target.abs().max().clamp_min(1e-300);a=(pred-target)/scale;b=target/scale
 return float(a.square().sum()/b.square().sum().clamp_min(1e-300))
