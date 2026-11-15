"""Exact binary/reciprocal formulation, convex epigraph, full global bounds."""
import time,json,hashlib,resource,itertools,threading,subprocess,sys,os
from pathlib import Path
import numpy as np
from pyscipopt import Model,quicksum,Eventhdlr,SCIP_EVENTTYPE

def objective(data,mask):
 N,Z,Nh,Zh=(data[n] for n in ('N','Z','Nh','Zh'))
 out=np.where(mask[None,:,None],N,Nh).sum(1)/np.where(mask[None,:],Z,Zh).sum(1)[:,None]
 od=N.sum(1)/Z.sum(1)[:,None]
 return float(np.square(out-od).sum()/(np.square(od).sum()+1e-12))

def setup(data,K,own):
 N,Z,Nh,Zh=(np.asarray(data[n],dtype=np.float64) for n in ('N','Z','Nh','Zh'))
 assert np.isfinite(N).all() and np.isfinite(Nh).all() and np.isfinite(Z).all() and np.isfinite(Zh).all() and (Z>0).all() and (Zh>0).all()
 Q,P,d=N.shape;assert Z.shape==(Q,P) and 1<=K<=P and 0<=own<P
 free=np.delete(np.arange(P),own);k=K-1;dn=N[:,free]-Nh[:,free];dz=Z[:,free]-Zh[:,free]
 a=N[:,own]+Nh[:,free].sum(1);b=Z[:,own]+Zh[:,free].sum(1)
 bounds=[]
 for descending in (False,True):
  idx=np.argsort(dz,axis=1);idx=idx[:,::-1] if descending else idx
  sel=np.zeros((Q,P),bool);sel[:,own]=True
  if k:sel[np.arange(Q)[:,None],free[idx[:,:k]]]=True
  bounds.append(np.where(sel,Z,Zh).sum(1))
 zmin,zmax=bounds;assert (zmin>0).all() and (zmax>=zmin).all()
 od=N.sum(1)/Z.sum(1)[:,None];cd=np.square(od).sum()+1e-12
 return free,a,b,dn,dz,1/zmax,1/zmin,od,cd

def solve(data,K,own,warm_masks,outdir,limit=900,checkpoints=(5,60,300,900)):
 root=Path(outdir);root.mkdir(parents=True,exist_ok=True);start=time.monotonic();cpu=time.process_time()
 free,a,b,dn,dz,lb,ub,od,cd=setup(data,K,own);Q,F,d=dn.shape;P=F+1
 hashes={n:hashlib.sha256(np.ascontiguousarray(data[n]).tobytes()).hexdigest() for n in ('N','Z','Nh','Zh')}
 masks=[np.asarray(z,dtype=bool) for z in warm_masks]
 for mask in masks:assert mask.shape==(P,) and mask.sum()==K and mask[own]
 warm_values=[objective(data,z) for z in masks];best=masks[int(np.argmin(warm_values))].copy()
 np.savez_compressed(root/'warm_starts.npz',masks=masks,objectives=warm_values)
 if K in (1,P):
  mask=np.ones(P,bool) if K==P else np.arange(P)==own
  value=objective(data,mask);res=dict(status='GLOBAL_TOL',analytic_unique_mask=True,LB=value,UB=value,absolute_gap=0.,relative_gap=0.,mask=np.where(mask)[0].tolist(),contribution_hashes=hashes)
  (root/'result.json').write_text(json.dumps(res,indent=2));return res
 objective_scale=1/max(min(warm_values),1e-6)
 m=Model('E11_full_denominator');m.setLogfile(str(root/'solver.log'));m.hideOutput()
 m.setRealParam('numerics/feastol',1e-8);m.setRealParam('limits/gap',5e-7);m.setRealParam('limits/absgap',objective_scale*5e-11)
 m.setBoolParam('nlp/disable',True);m.setIntParam('heuristics/subnlp/freq',-1);m.setIntParam('heuristics/multistart/freq',-1)
 m.setIntParam('randomization/randomseedshift',17);m.setIntParam('parallel/maxnthreads',8)
 # SCIP's main solve is serial unless concurrent solving is explicitly invoked.
 x=[m.addVar(f'x_{p}',vtype='B') for p in free];y=[m.addVar(f'y_{q}',lb=float(lb[q]),ub=float(ub[q])) for q in range(Q)]
 w=[[m.addVar(f'w_{q}_{p}',lb=0.,ub=float(ub[q])) for p in free] for q in range(Q)]
 t=[[m.addVar(f't_{q}_{j}',lb=None,ub=None) for j in range(d)] for q in range(Q)]
 ep=m.addVar('objective_epigraph',lb=0.);m.addCons(quicksum(x)==K-1)
 for q in range(Q):
  m.addCons(quicksum(w[q])==(K-1)*y[q]) # Valid cardinality-product tightening.
  for p in range(F):
   m.addCons(w[q][p]>=float(lb[q])*x[p]);m.addCons(w[q][p]<=float(ub[q])*x[p])
   m.addCons(w[q][p]>=y[q]-float(ub[q])*(1-x[p]));m.addCons(w[q][p]<=y[q]-float(lb[q])*(1-x[p]))
  m.addCons(float(b[q])*y[q]+quicksum(float(dz[q,p])*w[q][p] for p in range(F))==1.)
  for j in range(d):m.addCons(t[q][j]==float(a[q,j])*y[q]+quicksum(float(dn[q,p,j])*w[q][p] for p in range(F))-float(od[q,j]))
 eq=[m.addVar(f'query_error_{q}',lb=0.) for q in range(Q)]
 for q in range(Q):m.addCons(eq[q]>=objective_scale*quicksum(v*v for v in t[q])/float(cd))
 m.addCons(ep>=quicksum(eq));m.setObjective(ep,'minimize')
 # These are exact feasible products at each registered baseline mask.
 for mask,value in zip(masks,warm_values):
  sol=m.createSol();z=np.where(mask[None,:],data['Z'],data['Zh']).sum(1);ys=1/z
  out=np.where(mask[None,:,None],data['N'],data['Nh']).sum(1)/z[:,None]
  for p,v in enumerate(x):m.setSolVal(sol,v,int(mask[free[p]]))
  for q in range(Q):
   m.setSolVal(sol,y[q],float(ys[q]))
   for p,v in enumerate(w[q]):m.setSolVal(sol,v,float(ys[q]*mask[free[p]]))
   for j,v in enumerate(t[q]):m.setSolVal(sol,v,float(out[q,j]-od[q,j]))
  for q in range(Q):m.setSolVal(sol,eq[q],float(objective_scale*np.square(out[q]-od[q]).sum()/cd))
  m.setSolVal(sol,ep,objective_scale*value);m.addSol(sol,free=True)
 m.writeProblem(str(root/'model.cip'),verbose=False)
 model_hash=hashlib.sha256((root/'model.cip').read_bytes()).hexdigest()
 build_seconds=time.monotonic()-start;records=[];state={'LB':0.,'UB':min(warm_values),'solver_seconds':0.,'nodes':0,'source':'analytic nonnegative lower bound and verified warm start','mask':np.where(best)[0].tolist()}
 def publish():
  tmp=root/'latest_bound.tmp';tmp.write_text(json.dumps(state));os.replace(tmp,root/'latest_bound.json')
 publish()
 class Monitor(Eventhdlr):
  def eventinit(self):
   self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND|SCIP_EVENTTYPE.LPSOLVED|SCIP_EVENTTYPE.DUALBOUNDIMPROVED|SCIP_EVENTTYPE.NODEFOCUSED,self)
  def eventexec(self,event):
   lower=float(self.model.getDualbound())/objective_scale;upper=float(self.model.getPrimalbound())/objective_scale
   if upper<state['UB']:
    solution=self.model.getBestSol()
    if solution is not None:
     candidate=np.zeros(P,bool);candidate[own]=True;candidate[free]=[self.model.getSolVal(solution,v)>.5 for v in x]
     if candidate.sum()==K:
      true_upper=objective(data,candidate)
      if true_upper<state['UB']:state.update(UB=true_upper,mask=np.where(candidate)[0].tolist())
   state.update(LB=max(0.,lower),solver_seconds=float(self.model.getSolvingTime()),nodes=int(self.model.getNNodes()),source='latest SCIP event; timestamp records bound freshness')
   publish()
 m.includeEventhdlr(Monitor(),'checkpoint_monitor','publish valid bounds without interrupting LP solves')
 solve_start=time.monotonic();deadlines=sorted(set([min(float(s),limit) for s in checkpoints]+[float(limit)]))
 (root/'search_finished').unlink(missing_ok=True)
 monitor=subprocess.Popen([sys.executable,str(Path(__file__).with_name('checkpoint_monitor.py')),str(root),str(solve_start),json.dumps(deadlines)])
 m.setRealParam('limits/time',float(limit));m.optimize();(root/'search_finished').write_text('finished');monitor.wait(timeout=5)
 if (root/'checkpoints.json').exists():records=json.loads((root/'checkpoints.json').read_text())
 records.append(dict(final=True,solver_seconds=float(m.getSolvingTime()),LB=max(0.,float(m.getDualbound())/objective_scale),UB=float(m.getPrimalbound())/objective_scale,status=str(m.getStatus()),nodes=int(m.getNNodes())))
 (root/'checkpoints.json').write_text(json.dumps(records,indent=2))
 sol=m.getBestSol();lower=max(0.,float(m.getDualbound())/objective_scale);solver_status=str(m.getStatus());residual=None;integrality=None;solver_objective=None
 if sol is not None:
  xv=np.array([m.getSolVal(sol,v) for v in x]);mask=np.zeros(P,bool);mask[own]=True;mask[free]=xv>.5
  integrality=float(np.max(np.abs(xv-np.rint(xv))));ys=np.array([m.getSolVal(sol,v) for v in y]);ws=np.array([[m.getSolVal(sol,v) for v in row] for row in w]);ts=np.array([[m.getSolVal(sol,v) for v in row] for row in t])
  residual=float(max(np.max(np.abs(b*ys+(dz*ws).sum(1)-1)),np.max(np.abs(a*ys[:,None]+np.einsum('qpd,qp->qd',dn,ws)-od-ts)),np.max(np.abs(ws-ys[:,None]*xv[None,:]))))
  solver_objective=float(m.getObjVal())/objective_scale;upper=objective(data,mask);valid=mask.sum()==K and mask[own] and integrality<=1e-8 and residual<=1e-8 and abs(solver_objective-upper)<=max(1e-10,1e-6*abs(upper))
  if not valid:status='NUMERIC_UNRESOLVED'
  elif np.isfinite(lower) and lower>-1e19:status='GLOBAL_TOL' if upper-lower<=max(1e-10,1e-6*abs(upper)) and lower<=upper+1e-10 else 'BOUNDED_INCUMBENT'
  else:status='NO_FEASIBLE_CERTIFICATE'
 else:mask=best;upper=objective(data,mask);status='NO_FEASIBLE_CERTIFICATE'
 warm_start_only=any(np.array_equal(mask,z) for z in masks)
 if status=='BOUNDED_INCUMBENT' and warm_start_only and lower<=0.:status='NO_FEASIBLE_CERTIFICATE'
 if lower>upper+max(1e-10,1e-6*abs(upper)):status='NUMERIC_UNRESOLVED'
 res=dict(warm_start_only=warm_start_only,status=status,solver_status=solver_status,LB=lower,UB=upper,absolute_gap=upper-lower,relative_gap=(upper-lower)/max(abs(upper),1e-300),solver_objective=solver_objective,mask=np.where(mask)[0].tolist(),P=P,K=K,Q=Q,d=d,own=own,all_pages=True,model_hash=model_hash,contribution_hashes=hashes,solver_version=f'SCIP {m.getMajorVersion()}.{m.getMinorVersion()}.{m.getTechVersion()}',seed=17,runtime=float(m.getSolvingTime()),build_seconds=build_seconds,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,work=None,work_note='SCIP does not expose Gurobi Work units',node_count=int(m.getNNodes()),peak_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,feasibility_residual=residual,integrality_residual=integrality,denominator_condition_max=float(np.max(ub/lb)),objective_scale=objective_scale,configured_max_threads=8,execution='serial SCIP optimize',warm_objectives=warm_values)
 (root/'result.json').write_text(json.dumps(res,indent=2));return res

def exhaustive(data,K,own):
 P=data['Z'].shape[1];free=np.delete(np.arange(P),own);best=(float('inf'),None);count=0
 for choice in itertools.combinations(free,K-1):
  mask=np.zeros(P,bool);mask[own]=True;mask[list(choice)]=True;v=objective(data,mask);count+=1
  if v<best[0]:best=(v,mask)
 return best[0],best[1],count
