"""Compact PSD MIQP lower-bound model: solve bound is not original optimum."""
import argparse,time,json,hashlib
from pathlib import Path
import numpy as np
from pyscipopt import Model,quicksum,Eventhdlr,SCIP_EVENTTYPE
from .core import Problem
from .relaxation import build_h,qp_relax,soc_relax
ROOT=Path('/tmp/e11_compact')
def serial(r):return {k:v for k,v in r.items() if k!='point'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,required=True);a=ap.parse_args();root=ROOT/f'l{a.layer}';root.mkdir(exist_ok=True,parents=True);data=dict(np.load(f'/tmp/e11_results/contributions/horse_field_s101_c5_l{a.layer}_s0.npz'));p=Problem(data);loc=json.loads((root/'local_result.json').read_text());mask=np.zeros(410,bool);mask[loc['best_mask']]=True;ub=p.value(mask);scale=min(loc['baseline_mass'],loc['baseline_risk']);H,B,L,U,shift,guard,meta=build_h(p);np.savez_compressed(root/'lower_coefficients.npz',H=H,B=B,L=L,U=U,shift=shift,guard=guard)
 # Validate dominance on actual integer sets independently of any solver certificate.
 rng=np.random.default_rng(17);tests=[]
 for x in [data['mass'],data['risk'],mask]+[np.isin(np.arange(410),np.r_[p.own,rng.choice(p.free,40,replace=False)]) for _ in range(64)]:
  y=(~x).astype(float);q=float(y@H@y);E,z,j=p.state(x);soc=float(np.sum(np.square(E).sum(1)/(z*U))/p.C);true=p.value(x);assert q<=soc+1e-11 and soc<=true+1e-11;tests.append(dict(quadratic=q,soc=soc,true=true))
 (root/'lower_validation.json').write_text(json.dumps(dict(passed=True,tests=tests,**meta),indent=2))
 qp=qp_relax(p,H,B,shift,guard,initial=(~mask).astype(float));(root/'qp_relaxation.json').write_text(json.dumps(serial(qp),indent=2));print('QP_BOUND',a.layer,qp['LB'],ub,flush=True)
 m=Model('convex_MIQP_lower_bound');m.hideOutput();m.setRealParam('numerics/epsilon',1e-14);m.setRealParam('numerics/sumepsilon',1e-12);m.setLogfile(str(root/'miqp.log'));m.setRealParam('limits/time',900);m.setRealParam('limits/gap',1e-6);m.setRealParam('limits/absgap',1e-10/scale);m.setRealParam('numerics/feastol',1e-9);m.setIntParam('randomization/randomseedshift',17);m.setIntParam('parallel/maxnthreads',1);m.setBoolParam('constraints/nonlinear/reformbinprods',False);m.setBoolParam('constraints/nonlinear/assumeconvex',False);m.setBoolParam('nlp/disable',True);m.setIntParam('heuristics/subnlp/freq',-1);m.setIntParam('heuristics/multistart/freq',-1)
 free=p.free;y=[m.addVar(f'y_{i}',vtype='B') for i in free];ep=m.addVar('scaled_psd_objective',lb=0.);m.addCons(quicksum(y)==369)
 # Factorized PSD expression prevents SCIP from expanding each binary product.
 R=np.linalg.cholesky(B[np.ix_(free,free)]).T
 factor_error=float(369*np.linalg.norm(R.T@R-B[np.ix_(free,free)],ord=np.inf));guard+=factor_error
 v=[m.addVar(f'residual_{i}',lb=None,ub=None) for i in range(len(free))]
 for i in range(len(free)):m.addCons(v[i]==quicksum(float(R[i,j]/np.sqrt(scale))*y[j] for j in range(len(free)) if R[i,j]!=0.))
 m.addCons(ep>=quicksum(vi*vi for vi in v));m.setObjective(ep,'minimize')
 for x in (data['mass'],data['risk'],mask):
  sol=m.createSol();yy=(~x).astype(float)
  for binary_var,i in zip(y,free):m.setSolVal(sol,binary_var,float(yy[i]))
  vv=R@yy[free]/np.sqrt(scale)
  for vi,vi_value in zip(v,vv):m.setSolVal(sol,vi,float(vi_value))
  m.setSolVal(sol,ep,float(vv@vv)*(1+1e-10)+1e-12);assert m.checkSol(sol,printreason=True);m.addSol(sol,free=True)
 trace=[]
 class Monitor(Eventhdlr):
  def eventinit(self):self.model.catchEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED|SCIP_EVENTTYPE.LPSOLVED|SCIP_EVENTTYPE.BESTSOLFOUND,self)
  def eventexec(self,event):
   t=float(self.model.getSolvingTime())
   if not trace or t-trace[-1]['seconds']>=5:
    trace.append(dict(seconds=t,nodes=int(self.model.getNNodes()),LB=max(0.,float(self.model.getDualbound())*scale-shift*369-guard),surrogate_UB=float(self.model.getPrimalbound())*scale-shift*369));(root/'miqp_progress.json').write_text(json.dumps(trace,indent=2))
 m.includeEventhdlr(Monitor(),'bounds','bounds of the lower-bound model');m.writeProblem(str(root/'miqp.cip'),verbose=False);begin=time.process_time();m.optimize();rawlb=float(m.getDualbound())*scale-shift*369-guard;lb=max(qp['LB'],rawlb,0.)
 res=dict(model='PSD MIQP bound, factorized convex epigraph, 409 binary + 409 residual + 1 epigraph',factorization_guard=factor_error,status=str(m.getStatus()),solver_version=f'{m.getMajorVersion()}.{m.getMinorVersion()}.{m.getTechVersion()}',original_objective_LB=lb,solver_LB=max(0.,rawlb),continuous_tangent_LB=qp['LB'],surrogate_incumbent=float(m.getPrimalbound())*scale-shift*369,runtime=m.getSolvingTime(),cpu_seconds=time.process_time()-begin,nodes=m.getNNodes(),original_incumbent_UB=ub,original_absolute_gap=ub-lb,original_relative_gap=(ub-lb)/ub,scale=scale,**meta,model_sha256=hashlib.sha256((root/'miqp.cip').read_bytes()).hexdigest())
 if m.getBestSol() is not None:
  yy=np.zeros(410,bool);yy[free]=[m.getSolVal(m.getBestSol(),v)>.5 for v in y];candidate=~yy
  if candidate.sum()==41 and candidate[p.own]:res['surrogate_mask_true_objective']=p.value(candidate);res['surrogate_mask']=np.where(candidate)[0].tolist()
 assert lb<=ub+max(1e-10,1e-6*ub);(root/'miqp_result.json').write_text(json.dumps(res,indent=2));print('MIQP',a.layer,lb,ub,flush=True)
 if ub-lb>max(1e-10,.05*ub):
  soc=soc_relax(p,U,guard,initial=qp['point']);assert soc['LB']<=ub+1e-10;(root/'soc_relaxation.json').write_text(json.dumps(serial(soc),indent=2));np.save(root/'soc_point.npy',soc['point']);print('SOC_TANGENT',a.layer,soc['LB'],ub,flush=True)
if __name__=='__main__':main()
