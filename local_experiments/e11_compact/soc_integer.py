"""Rotated SOC integer lower-bound model; never label it the true objective."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from pyscipopt import Model,quicksum,sqrt,Eventhdlr,SCIP_EVENTTYPE
from .core import Problem
ROOT=Path('/tmp/e11_compact')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,required=True);a=ap.parse_args();root=ROOT/f'l{a.layer}';p=Problem(dict(np.load(f'/tmp/e11_results/contributions/horse_field_s101_c5_l{a.layer}_s0.npz')));loc=json.loads((root/'local_result.json').read_text());coeff=dict(np.load(root/'lower_coefficients.npz'));L,U=coeff['L'],coeff['U'];scale=min(loc['baseline_mass'],loc['baseline_risk']);guard=float(coeff['guard']);x=np.zeros(410,bool);x[loc['best_mask']]=True
 m=Model('integer_SOC_lower_bound');m.hideOutput();m.setRealParam('numerics/epsilon',1e-14);m.setRealParam('numerics/sumepsilon',1e-12);m.setLogfile(str(root/'soc_integer.log'));m.setRealParam('limits/time',900);m.setRealParam('limits/gap',1e-6);m.setRealParam('limits/absgap',1e-10/scale);m.setRealParam('numerics/feastol',1e-10);m.setIntParam('randomization/randomseedshift',17);m.setIntParam('parallel/maxnthreads',1);m.setBoolParam('constraints/nonlinear/reformbinprods',False);m.setBoolParam('nlp/disable',True);m.setIntParam('heuristics/subnlp/freq',-1);m.setIntParam('heuristics/multistart/freq',-1)
 y=[m.addVar(f'y_{i}',vtype='B') for i in p.free];z=[m.addVar(f'z_{q}',lb=float(L[q]),ub=float(U[q])) for q in range(64)];t=[m.addVar(f't_{q}',lb=0.) for q in range(64)];u=[[m.addVar(f'e_{q}_{j}',lb=None,ub=None) for j in range(128)] for q in range(64)];m.addCons(quicksum(y)==369)
 for q in range(64):
  m.addCons(z[q]==1-quicksum(float(p.d[q,i])*v for i,v in zip(p.free,y)))
  for j in range(128):m.addCons(u[q][j]==quicksum(float(p.e[q,i,j])*v for i,v in zip(p.free,y)))
  # ||(2E,t-z)|| <= t+z, with t,z >=0: exactly E'E <= t*z.
  m.addCons(sqrt(4*quicksum(v*v for v in u[q])+(t[q]-z[q])**2)<=t[q]+z[q])
 obj=quicksum(t[q]/float(p.C*U[q]*scale) for q in range(64));m.setObjective(obj,'minimize')
 prior=json.loads((root/'qp_relaxation.json').read_text())['LB']
 # Original MIQP-model LB also bounds this tighter SOC model. Valid strengthening.
 m.addCons(obj>=max(0.,prior)/scale)
 warm_checks=[]
 for mask in (p.data['mass'],p.data['risk'],x):
  yy=(~mask).astype(float);E=np.einsum('qpd,p->qd',p.e,yy,optimize=True);zz=1-p.d@yy;tt=np.square(E).sum(1)/zz*(1+1e-10)+1e-12;sol=m.createSol()
  for v,i in zip(y,p.free):m.setSolVal(sol,v,float(yy[i]))
  for q in range(64):
   m.setSolVal(sol,z[q],float(zz[q]));m.setSolVal(sol,t[q],float(tt[q]))
   for j in range(128):m.setSolVal(sol,u[q][j],float(E[q,j]))
  valid=bool(m.checkSol(sol,printreason=True));accepted=bool(m.addSol(sol,free=True));warm_checks.append(dict(valid=valid,accepted=accepted));assert valid,warm_checks
 (root/'soc_warm_checks.json').write_text(json.dumps(warm_checks,indent=2))
 trace=[]
 class Monitor(Eventhdlr):
  def eventinit(self):self.model.catchEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED|SCIP_EVENTTYPE.LPSOLVED,self)
  def eventexec(self,event):
   now=float(self.model.getSolvingTime())
   if not trace or now-trace[-1]['seconds']>=5:trace.append(dict(seconds=now,LB=max(prior,float(self.model.getDualbound())*scale-guard),nodes=int(self.model.getNNodes())));(root/'soc_integer_progress.json').write_text(json.dumps(trace,indent=2))
 m.includeEventhdlr(Monitor(),'lower_bounds','SOC relaxation global bounds');m.writeProblem(str(root/'soc_integer.cip'),verbose=False);begin=time.process_time();m.optimize();numeric_guard=guard+10e-10*float(np.sum(1/(p.C*U*L)));lb=max(prior,float(m.getDualbound())*scale-numeric_guard,0.)
 result=dict(model='integer SOC lower-bound model; 64 rotated cones; 409 binary, no Query-Page products',solver_status=str(m.getStatus()),LB=lb,original_incumbent_UB=loc['best_objective'],absolute_gap=loc['best_objective']-lb,relative_gap=(loc['best_objective']-lb)/loc['best_objective'],runtime=m.getSolvingTime(),cpu_seconds=time.process_time()-begin,nodes=m.getNNodes(),numerical_guard=numeric_guard,surrogate_incumbent=float(m.getPrimalbound())*scale,model_hash=hashlib.sha256((root/'soc_integer.cip').read_bytes()).hexdigest())
 if m.getBestSol() is not None:
  yy=np.zeros(410,bool);yy[p.free]=[m.getSolVal(m.getBestSol(),v)>.5 for v in y];mask=~yy
  assert mask.sum()==41 and mask[p.own];result['candidate_true_J']=p.value(mask);result['candidate_mask']=np.where(mask)[0].tolist()
 assert lb<=loc['best_objective']+max(1e-10,1e-6*loc['best_objective']);(root/'soc_integer_result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
