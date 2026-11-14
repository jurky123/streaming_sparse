"""Full-domain branch-and-bound with node-valid denominator bounds and convex tangents."""
import argparse,json,time,heapq,hashlib
from pathlib import Path
import numpy as np
from .problem import ResidualProblem
from ..e11_compact.relaxation import build_h,qp_relax,soc_relax,linear_min,soc_projected
ROOT=Path('/tmp/set_oracle_v1')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--event',required=True);ap.add_argument('--seconds',type=float,default=90);a=ap.parse_args();root=ROOT/'global_bounds'/a.event;root.mkdir(exist_ok=True,parents=True);p=ResidualProblem(ROOT/'contributions'/a.event);local=json.loads((ROOT/'searches'/'v1'/a.event/'result.json').read_text());best=np.zeros(p.P,bool);best[local['best_mask']]=True;ub=p.value(best);global_root_lb=0.;socfile=root/'soc_integer_result.json'
 if socfile.exists():
  s=json.loads(socfile.read_text());global_root_lb=max(global_root_lb,0.) # Solver bounds kept separate; this proof uses only independently computed tangents.
  if s.get('candidate_true_J',np.inf)<ub:best[:]=False;best[s['candidate_mask']]=True;ub=p.value(best)
 fixed={p.own:0};H,B,L,U,shift,guard,meta=build_h(p);relax=qp_relax(p,H,B,shift,guard,initial=(~best).astype(float));point=relax['point'];global_root_lb=max(global_root_lb,relax['LB']);g=2*B@point;intercept=float(point@B@point-g@point-shift*(p.P-p.K)-guard)
 # Continuous SOC tangent is a separately valid bound, including for layers not requiring integer SOC.
 soc=soc_relax(p,U,guard,initial=point);(root/'soc_relaxation.json').write_text(json.dumps({k:v for k,v in soc.items() if k!='point'},indent=2));np.save(root/'soc_point.npy',soc['point']);global_root_lb=max(global_root_lb,soc['LB'])
 serial=0;heap=[(max(global_root_lb,relax['LB']),serial,fixed,point,g,intercept)];start=time.monotonic();cpu=time.process_time();expanded=pruned=solved=0;fathomed_floor=float('inf');max_depth=0;record=root/'branch_nodes.jsonl';journal=record.open('w');journal.write(json.dumps(dict(id=0,parent=None,LB=heap[0][0],fixed=fixed,point=point.tolist(),gradient=g.tolist(),intercept=intercept))+'\n');best_updates=[]
 while heap and time.monotonic()-start<a.seconds:
  tol=max(1e-10,1e-6*abs(ub))
  if heap[0][0]>=ub-tol:break
  lb,ident,fixed,point,grad,intercept=heapq.heappop(heap);expanded+=1;free=np.array([i for i in range(p.P) if i not in fixed]);remaining=(p.P-p.K)-sum(fixed.values())
  frac=np.minimum(point[free],1-point[free]);uncert=np.std(p.d[:,free],axis=0);score=frac+1e-5*uncert/max(float(uncert.max()),1e-20);var=int(free[np.argmax(score)])
  # Both children are processed before checking time: the remaining frontier always covers all unpruned masks.
  for value in (0,1):
   serial+=1;child=dict(fixed);child[var]=value;remaining=(p.P-p.K)-sum(child.values());cfree=np.array([i for i in range(p.P) if i not in child]);max_depth=max(max_depth,len(child)-1)
   rec=dict(id=serial,parent=ident,branch_page=var,summary_bit=value,depth=len(child)-1)
   if remaining<0 or remaining>len(cfree):rec['pruned']='cardinality_infeasible';pruned+=1;journal.write(json.dumps(rec)+'\n');continue
   tangent_min,_=linear_min(grad,child,(p.P-p.K));cheap=max(lb,global_root_lb,intercept+tangent_min)
   if cheap>=ub-tol:
    fathomed_floor=min(fathomed_floor,cheap);rec.update(pruned='valid_parent_tangent',LB=cheap,UB=ub);pruned+=1;journal.write(json.dumps(rec)+'\n');continue
   if remaining in (0,len(cfree)):
    yy=np.zeros(p.P,bool)
    for i,v in child.items():yy[i]=v
    if remaining:yy[cfree]=True
    x=~yy;j=p.value(x);fathomed_floor=min(fathomed_floor,j);rec.update(pruned='unique_feasible_mask_evaluated',objective=j,mask=np.where(x)[0].tolist());solved+=1
    if j<ub:best=x;ub=j;best_updates.append(dict(node=serial,objective=j,mask=np.where(best)[0].tolist()))
    journal.write(json.dumps(rec)+'\n');continue
   Hc,Bc,Lc,Uc,sh,gc,mc=build_h(p,child);q=qp_relax(p,Hc,Bc,sh,gc,child,initial=point,maxiter=350);qs=soc_projected(p,Uc,gc,child,q['point'],maxiter=100,L=Lc);use_soc=qs['LB']>q['LB'];q=qs if use_soc else q;bound=max(cheap,q['LB']);pt=q['point']
   # Rounding provides a global feasible upper bound, never a lower bound or pruning argument.
   yy=np.zeros(p.P,bool)
   for i,v in child.items():yy[i]=v
   if remaining:yy[cfree[np.argsort(pt[cfree])[-remaining:]]]=True
   x=~yy;j=p.value(x)
   if j<ub:best=x;ub=j;best_updates.append(dict(node=serial,objective=j,mask=np.where(best)[0].tolist()))
   tol=max(1e-10,1e-6*abs(ub));rec.update(LB=bound,UB=ub,denominator_U_min=float(Uc.min()),denominator_U_max=float(Uc.max()),roundoff_guard=gc)
   gg=q['gradient'] if use_soc else 2*Bc@pt;cc=q['intercept'] if use_soc else float(pt@Bc@pt-gg@pt-sh*(p.P-p.K)-gc);rec.update(plane_kind='secant' if use_soc else 'qp',fixed=child,point=pt.tolist(),gradient=gg.tolist(),intercept=cc)
   if bound>=ub-tol:rec['pruned']='node_convex_tangent';pruned+=1;fathomed_floor=min(fathomed_floor,bound)
   else:
    heapq.heappush(heap,(bound,serial,child,pt,gg,cc));rec.update(fixed=child,point=pt.tolist(),gradient=gg.tolist(),intercept=cc)
   journal.write(json.dumps(rec)+'\n')
  if expanded%10==0:
   journal.flush();lower=min(ub,fathomed_floor,heap[0][0] if heap else float('inf'));(root/'branch_progress.json').write_text(json.dumps(dict(expanded=expanded,pruned=pruned,frontier=len(heap),LB=lower,UB=ub,relative_gap=(ub-lower)/ub,seconds=time.monotonic()-start),indent=2));print('BRANCH',a.event,expanded,len(heap),lower,ub,flush=True)
 journal.close();lower=min(ub,fathomed_floor,heap[0][0] if heap else float('inf'));tol=max(1e-10,1e-6*abs(ub));status='GLOBAL_TOL' if ub-lower<=tol else 'BOUNDED_INCUMBENT'
 assert best.sum()==p.K and best[p.own];assert abs(p.state(best)[2]-p.value(best))<1e-11
 result=dict(status=status,LB=lower,UB=ub,absolute_gap=ub-lower,relative_gap=(ub-lower)/ub,mask=np.where(best)[0].tolist(),expanded_nodes=expanded,pruned_nodes=pruned,unique_leaves=solved,frontier_nodes=len(heap),max_depth=max_depth,runtime=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,root_LB=global_root_lb,pruned_floor=fathomed_floor,solver_bounds_used_for_pruning=False,best_updates=best_updates,all_legal_pages=True,full_query_count=64,numerical_tolerance=tol,certificate_scope='Original variable-denominator FP64 objective; full binary domain; node-valid convex supporting hyperplanes',node_journal_sha256=hashlib.sha256(record.read_bytes()).hexdigest())
 (root/'branch_frontier.json').write_text(json.dumps([dict(id=t[1],LB=t[0],fixed=t[2]) for t in heap],indent=2));(root/'branch_result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
