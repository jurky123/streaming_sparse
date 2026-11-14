"""Convex relaxations with globally valid supporting-hyperplane lower bounds."""
import numpy as np,time
from scipy.optimize import minimize

def linear_min(g,fixed,total):
 free=np.array([i for i in range(len(g)) if i not in fixed]);ones=[i for i,v in fixed.items() if v];remaining=total-len(ones);assert 0<=remaining<=len(free)
 v=np.zeros(len(g));v[ones]=1
 if remaining:v[free[np.argsort(g[free])[:remaining]]]=1
 return float(g@v),v

def project(x,fixed,total):
 x=x.copy();free=np.array([i for i in range(len(x)) if i not in fixed]);m=total-sum(fixed.values())
 for i,v in fixed.items():x[i]=v
 if not len(free):return x
 if m==0:x[free]=0;return x
 if m==len(free):x[free]=1;return x
 lo=float(x[free].min()-1);hi=float(x[free].max())
 for _ in range(48):
  mid=(lo+hi)/2
  if np.clip(x[free]-mid,0,1).sum()>m:lo=mid
  else:hi=mid
 x[free]=np.clip(x[free]-(lo+hi)/2,0,1);return x

def build_h(p,fixed=None):
 fixed={p.own:0} if fixed is None else dict(fixed)|{p.own:0};L,U=p.bounds(fixed);H=np.einsum('q,qpr->pr',1/(p.C*U**2),p.G,optimize=True);H=(H+H.T)/2
 eig=np.linalg.eigvalsh(H);shift=max(0.,-float(eig[0]))+10*np.finfo(float).eps*len(H)*max(float(eig[-1]),1e-12);B=H+shift*np.eye(p.P)
 # A PSD shift adds exactly shift*369 at every binary feasible y; subtract that constant.
 # Conservative accumulated floating point guard for dense Gram construction/evaluation.
 if not hasattr(p,'absG'):
  absolute=np.abs(p.e);p.absG=absolute@absolute.transpose(0,2,1)
 absG=p.absG;absH=np.einsum('q,qpr->pr',1/(p.C*U**2),absG,optimize=True);u=np.finfo(float).eps/2;gamma=(128+64+20)*u/(1-(128+64+20)*u)
 guard=max(1e-12,float((p.P-p.K)*gamma*np.max(absH.sum(1))))
 return H,B,L,U,shift,guard,dict(min_eigenvalue=float(eig[0]),max_eigenvalue=float(eig[-1]),psd_shift=shift,roundoff_guard=guard)

def qp_relax(p,H,B,shift,guard,fixed=None,initial=None,maxiter=2000):
 start=time.monotonic();fixed={p.own:0} if fixed is None else dict(fixed)|{p.own:0};m=p.P-p.K;y=project(np.full(p.P,m/p.P) if initial is None else initial,fixed,m);z=y.copy();t=1.;lip=2*np.linalg.norm(B,ord=np.inf);best=-float('inf');besty=y.copy();trace=[]
 for i in range(maxiter):
  g=2*(B@y);f=float(y@(B@y));linear,_=linear_min(g,fixed,m);lower=f+linear-g@y-shift*m-guard
  if lower>best:best=float(lower);besty=y.copy()
  if i%100==0:trace.append(dict(iteration=i,LB=max(0.,best),relaxation_value=f-shift*m))
  if f-shift*m-best<=max(1e-12,abs(f)*1e-8):break
  nexty=project(z-2*(B@z)/lip,fixed,m);nextt=(1+np.sqrt(1+4*t*t))/2;z=nexty+(t-1)/nextt*(nexty-y);y=nexty;t=nextt
 return dict(LB=max(0.,best),relaxation_value=float(besty@(B@besty)-shift*m),iterations=i+1,wall_seconds=time.monotonic()-start,trace=trace,point=besty)

def soc_relax(p,U,guard,fixed=None,initial=None,maxiter=300):
 start=time.monotonic();fixed={p.own:0} if fixed is None else dict(fixed)|{p.own:0};m=p.P-p.K;free=np.array([i for i in range(p.P) if i not in fixed]);fixed_y=np.zeros(p.P)
 for i,v in fixed.items():fixed_y[i]=v
 total=m-sum(fixed.values());scale=min(p.value(p.data['mass']),p.value(p.data['risk']));best=[-float('inf'),None,None];trace=[];calls=[0]
 def full(x):v=fixed_y.copy();v[free]=x;return v
 def eval_fg(x):
  y=full(x);E=np.einsum('qpd,p->qd',p.e,y,optimize=True);z=1-p.d@y;assert (z>0).all();n=np.square(E).sum(1);f=float(np.sum(n/(z*U))/p.C)
  g=(2*np.einsum('qpd,qd,q->p',p.e,E,1/(z*U),optimize=True)+np.einsum('qp,q->p',p.d,n/(z*z*U)))/p.C
  linear,_=linear_min(g,fixed,m);lb=float(f+linear-g@y-guard)
  if lb>best[0]:best[:]=[lb,y.copy(),f]
  calls[0]+=1
  if calls[0]%20==0:trace.append(dict(evaluations=calls[0],LB=max(0.,best[0]),relaxation_value=f))
  return f/scale,g[free]/scale
 y0=project(np.full(p.P,m/p.P) if initial is None else initial,fixed,m)
 opt=minimize(eval_fg,y0[free],jac=True,method='SLSQP',bounds=[(0,1)]*len(free),constraints=[dict(type='eq',fun=lambda x:x.sum()-total,jac=lambda x:np.ones_like(x))],options=dict(maxiter=maxiter,ftol=1e-11,disp=False))
 eval_fg(opt.x)
 return dict(LB=max(0.,best[0]),relaxation_value=best[2],point=best[1],iterations=int(opt.nit),optimizer_success=bool(opt.success),optimizer_message=str(opt.message),wall_seconds=time.monotonic()-start,trace=trace,bound_source='supporting hyperplane minimized over full cardinality polytope, valid regardless of local optimizer status')

def soc_projected(p,U,guard,fixed,initial,maxiter=100,L=None):
 """Fast convex perspective descent; only supporting planes certify bounds."""
 if not hasattr(p,'Eflat'):p.Eflat=np.ascontiguousarray(p.e.transpose(0,2,1).reshape(p.Q*p.D,p.P))
 A=p.Eflat;m=p.P-p.K;best=-float('inf');besty=initial.copy();bestg=None;bestc=None;lip=None
 def fg(y):
  E=(A@y).reshape(p.Q,p.D);z=1-p.d@y
  if np.any(z<=0):return float('inf'),None
  n=np.square(E).sum(1);coef=U if L is None else L+U;den=z*coef if L is None else z*coef-L*U
  if np.any(den<=0):return float('inf'),None
  f=float(np.sum(n/den)/p.C);g=(2*A.T@(E/den[:,None]).reshape(-1)+p.d.T@(n*coef/den**2))/p.C
  return f,g
 y=project(initial,fixed,m);v=y.copy();t=1.
 for it in range(maxiter):
  f,g=fg(v)
  if g is None:v=y.copy();t=1.;f,g=fg(v)
  mn,_=linear_min(g,fixed,m);c=float(f-g@v-guard);lb=c+mn
  if lb>best:best=lb;besty=v.copy();bestg=g.copy();bestc=c
  if lip is None:lip=max(float(np.linalg.norm(g,np.inf)),1e-8)
  for _ in range(60):
   nxt=project(v-g/lip,fixed,m);nf,_=fg(nxt);diff=nxt-v
   if nf<=f+g@diff+.5*lip*(diff@diff)+1e-15:break
   lip*=2
  nt=(1+np.sqrt(1+4*t*t))/2;v=nxt+(t-1)/nt*(nxt-y);y=nxt;t=nt;lip*=.9
 return dict(LB=max(0.,best),point=besty,gradient=bestg,intercept=bestc,iterations=maxiter)
