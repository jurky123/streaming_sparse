from ..e30_beta_diagnostics.run import *
from ..e25_decision_diagnostics.report import aggregate
import csv
E30=DEST;DEST=ROOT/'artifacts/e38_action_attribution_v1'
EPS=1e-12
METHODS=['E29','TrueBeta']+[f'G{g}-{w}' for g in [2,4] for w in ['mass','direction','jacobian']]
def pick(sc):return int(sc.argmax()) if len(sc) and float(sc.max())>0 else -1
def value(sc,j):return 0. if j<0 else float(sc[j])
def endpoints(j,r,a):return [] if j<0 else [int(r[j]),int(a[j])]
def mix(rp,beta,lm,cen,mask):
 v=cen+rp;alpha=(lm+beta).softmax(1);omega=torch.where(mask[None],lm+beta,lm).softmax(1);D=(alpha[...,None]*v).sum(1);H=(omega[...,None]*(cen+mask[None,:,None]*rp)).sum(1);return D,H,alpha,omega

def proposal_parts(rp,beta,lm,cen,mask,r,a,j,norm):
 if j<0:return dict(I=0.,B=0.,A=0.,rho=0.,pi_positive=0.,gain=0.,necessary_condition=True)
 D,H,_,_=mix(rp,beta,lm,cen,mask);mm=mask.clone();mm[r[j]]=False;mm[a[j]]=True;_,new,_,_=mix(rp,beta,lm,cen,mm);E=D-H;delta=new-H;ii=2*(E*delta).sum(-1);bb=delta.square().sum(-1);I=float(ii.mean()/norm);B=float(bb.mean()/norm);A=float(E.square().sum(-1).mean()/norm)
 return dict(I=I,B=B,A=A,rho=I/(B+EPS),pi_positive=float(((ii-bb)>0).double().mean()),gain=I-B,necessary_condition=bool(I-B<=1e-12 or B<4*A+1e-12))
def role_beta(rp,beta,true,lm,cen,mask,C,groups,kind):
 if kind=='mass':lw=2*lm
 elif kind=='direction':lw=2*lm+((rp+cen).square().sum(-1)+EPS).log()
 else:
  D,H,alpha,omega=mix(rp,beta,lm,cen,mask);v=cen+rp;jac=alpha[...,None]*(v-D[:,None])-mask[None,:,None]*omega[...,None]*(v-H[:,None]);lw=(jac.square().sum(-1)+EPS).log()
 labels=[C,~C] if groups==2 else [C&mask,C&~mask,~C&mask,~C&~mask];out=beta.clone();info=[]
 for k,ix in enumerate(labels):
  if not ix.any():info.append(dict(group=k,pages=0,max_weight=0.,effective_pages=0.,offset_mean=0.));continue
  w=lw[:,ix].softmax(1);b=(w*(true[:,ix]-beta[:,ix])).sum(1);out[:,ix]+=b[:,None];info.append(dict(group=k,pages=int(ix.sum()),max_weight=float(w.max(1).values.mean()),effective_pages=float((1/w.square().sum(1)).mean()),offset_mean=float(b.mean()),offset_abs_max=float(b.abs().max())))
 assert torch.stack(labels).sum(0).eq(1).all();return out,info

def pair_scores(rp,beta,lm,cen,mask,r,a,pair,norm):
 jj=[j for j in pair if j>=0]
 if not jj:return [0.,0.]
 sc,_=local_scores(rp,beta,lm,cen,mask,r[jj],a[jj],norm);vals=iter(sc.tolist());return [next(vals) if j>=0 else 0. for j in pair]
def margin(*args):
 vv=pair_scores(*args);return vv[0]-vv[1]
def true_pair(c,d,mask,r,a,pair,qs,B,norm,perp=False):
 N,Z,OD=d['N'][qs],d['Z'][qs],d['dense'][qs];nh,zh=c['nh'][qs],c['zh'][qs];ns=torch.where(mask[None,:,None],N,nh).sum(1);zs=torch.where(mask[None],Z,zh).sum(1);H=ns/zs[:,None];E=OD-H;ans=[]
 for j in pair:
  if j<0:ans.append(0.);continue
  delta=(ns-(N-nh)[:,r[j]]+(N-nh)[:,a[j]])/(zs-(Z-zh)[:,r[j]]+(Z-zh)[:,a[j]])[:,None]-H
  if B is None:ee,dd=E,delta
  elif perp:ee=E-(E@B)@B.T;dd=delta-(delta@B)@B.T
  else:ee,dd=E@B,delta@B
  ans.append(float((2*(ee*dd).sum(-1)-dd.square().sum(-1)).mean()/norm))
 return ans

def shapley3(values):
 result=[]
 for j in range(3):
  ans=0.
  for bits in range(8):
   if bits&(1<<j):continue
   k=bits.bit_count();weight=math.factorial(k)*math.factorial(2-k)/6;ans+=weight*(values[bits]-values[bits|(1<<j)])
  result.append(ans)
 return result
