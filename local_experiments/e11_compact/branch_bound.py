"""Full-domain branch-and-bound with node-valid denominator bounds and convex tangents."""
import argparse,json,time,heapq,hashlib
from pathlib import Path
import numpy as np
from .core import Problem
from .relaxation import build_h,qp_relax,soc_relax,linear_min
ROOT=Path('/tmp/e11_compact')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,required=True);ap.add_argument('--seconds',type=float,default=900);ap.add_argument('--resume',action='store_true');a=ap.parse_args();root=ROOT/f'l{a.layer}';data=dict(np.load(f'/tmp/e11_results/contributions/horse_field_s101_c5_l{a.layer}_s0.npz'));p=Problem(data);local=json.loads((root/'local_result.json').read_text());best=np.zeros(410,bool);best[local['best_mask']]=True;ub=p.value(best);miqp=json.loads((root/'miqp_result.json').read_text());global_root_lb=0.;socfile=root/'soc_integer_result.json'
 if socfile.exists():
  s=json.loads(socfile.read_text());global_root_lb=max(global_root_lb,0.) # Solver bounds kept separate; this proof uses only independently computed tangents.
  if s.get('candidate_true_J',np.inf)<ub:best[:]=False;best[s['candidate_mask']]=True;ub=p.value(best)
 fixed={p.own:0};H,B,L,U,shift,guard,meta=build_h(p);relax=qp_relax(p,H,B,shift,guard,initial=(~best).astype(float));point=relax['point'];global_root_lb=max(global_root_lb,relax['LB']);g=2*B@point;intercept=float(point@B@point-g@point-shift*369-guard)
 # Continuous SOC tangent is a separately valid bound, including for layers not requiring integer SOC.
 soc=soc_relax(p,U,guard,initial=point);(root/'soc_relaxation.json').write_text(json.dumps({k:v for k,v in soc.items() if k!='point'},indent=2));np.save(root/'soc_point.npy',soc['point']);global_root_lb=max(global_root_lb,soc['LB'])
 serial=0;heap=[(max(global_root_lb,relax['LB']),serial,fixed,point,g,intercept)];start=time.monotonic();cpu=time.process_time();expanded=pruned=solved=0;fathomed_floor=float('inf');max_depth=0;record=root/'branch_nodes.jsonl';prior_runtime=prior_cpu=0.;best_updates=[]
 if a.resume:
  previous=json.loads((root/'branch_result.json').read_text());oldrows={r['id']:r for r in map(json.loads,record.read_text().splitlines())};frontier=json.loads((root/'branch_frontier.json').read_text());heap=[]
  for node in frontier:
   rr=oldrows[node['id']];heap.append((node['LB'],node['id'],{int(k):v for k,v in node['fixed'].items()},np.array(rr['point']),np.array(rr['gradient']),rr['intercept']))
  heapq.heapify(heap);serial=max(oldrows);expanded=previous['expanded_nodes'];pruned=previous['pruned_nodes'];solved=previous['unique_leaves'];fathomed_floor=previous['pruned_floor'];max_depth=previous['max_depth'];prior_runtime=previous['runtime'];prior_cpu=previous['cpu_seconds'];best_updates=previous['best_updates'];best[:]=False;best[previous['mask']]=True;ub=p.value(best);global_root_lb=previous['root_LB'];journal=record.open('a')
 else:
  journal=record.open('w');journal.write(json.dumps(dict(id=0,parent=None,LB=heap[0][0],fixed=fixed,point=point.tolist(),gradient=g.tolist(),intercept=intercept))+'\n')
 while heap and time.monotonic()-start<a.seconds:
  tol=max(1e-10,1e-6*abs(ub))
  if heap[0][0]>=ub-tol:break
  lb,ident,fixed,point,grad,intercept=heapq.heappop(heap);expanded+=1;free=np.array([i for i in range(410) if i not in fixed]);remaining=369-sum(fixed.values())
  frac=np.minimum(point[free],1-point[free]);uncert=np.std(p.d[:,free],axis=0);score=frac+1e-5*uncert/max(float(uncert.max()),1e-20);var=int(free[np.argmax(score)])
  # Both children are processed before checking time: the remaining frontier always covers all unpruned masks.
  for value in (0,1):
   serial+=1;child=dict(fixed);child[var]=value;remaining=369-sum(child.values());cfree=np.array([i for i in range(410) if i not in child]);max_depth=max(max_depth,len(child)-1)
   rec=dict(id=serial,parent=ident,branch_page=var,summary_bit=value,depth=len(child)-1)
   if remaining<0 or remaining>len(cfree):rec['pruned']='cardinality_infeasible';pruned+=1;journal.write(json.dumps(rec)+'\n');continue
   tangent_min,_=linear_min(grad,child,369);cheap=max(lb,global_root_lb,intercept+tangent_min)
   if cheap>=ub-tol:
    fathomed_floor=min(fathomed_floor,cheap);rec.update(pruned='valid_parent_tangent',LB=cheap,UB=ub);pruned+=1;journal.write(json.dumps(rec)+'\n');continue
   if remaining in (0,len(cfree)):
    yy=np.zeros(410,bool)
    for i,v in child.items():yy[i]=v
    if remaining:yy[cfree]=True
    x=~yy;j=p.value(x);fathomed_floor=min(fathomed_floor,j);rec.update(pruned='unique_feasible_mask_evaluated',objective=j,mask=np.where(x)[0].tolist());solved+=1
    if j<ub:best=x;ub=j;best_updates.append(dict(node=serial,objective=j,mask=np.where(best)[0].tolist()))
    journal.write(json.dumps(rec)+'\n');continue
   Hc,Bc,Lc,Uc,sh,gc,mc=build_h(p,child);q=qp_relax(p,Hc,Bc,sh,gc,child,initial=point,maxiter=350);bound=max(cheap,q['LB']);pt=q['point']
   # Rounding provides a global feasible upper bound, never a lower bound or pruning argument.
   yy=np.zeros(410,bool)
   for i,v in child.items():yy[i]=v
   if remaining:yy[cfree[np.argsort(pt[cfree])[-remaining:]]]=True
   x=~yy;j=p.value(x)
   if j<ub:best=x;ub=j;best_updates.append(dict(node=serial,objective=j,mask=np.where(best)[0].tolist()))
   tol=max(1e-10,1e-6*abs(ub));rec.update(LB=bound,UB=ub,denominator_U_min=float(Uc.min()),denominator_U_max=float(Uc.max()),roundoff_guard=gc)
   gg=2*Bc@pt;cc=float(pt@Bc@pt-gg@pt-sh*369-gc);rec.update(fixed=child,point=pt.tolist(),gradient=gg.tolist(),intercept=cc)
   if bound>=ub-tol:rec['pruned']='node_convex_tangent';pruned+=1;fathomed_floor=min(fathomed_floor,bound)
   else:
    gg=2*Bc@pt;cc=float(pt@Bc@pt-gg@pt-sh*369-gc);heapq.heappush(heap,(bound,serial,child,pt,gg,cc));rec.update(fixed=child,point=pt.tolist(),gradient=gg.tolist(),intercept=cc)
   journal.write(json.dumps(rec)+'\n')
  if expanded%10==0:
   journal.flush();lower=min(ub,fathomed_floor,heap[0][0] if heap else float('inf'));(root/'branch_progress.json').write_text(json.dumps(dict(expanded=expanded,pruned=pruned,frontier=len(heap),LB=lower,UB=ub,relative_gap=(ub-lower)/ub,seconds=time.monotonic()-start),indent=2));print('BRANCH',a.layer,expanded,len(heap),lower,ub,flush=True)
 journal.close();lower=min(ub,fathomed_floor,heap[0][0] if heap else float('inf'));tol=max(1e-10,1e-6*abs(ub));status='GLOBAL_TOL' if ub-lower<=tol else 'BOUNDED_INCUMBENT'
 assert best.sum()==41 and best[p.own];assert abs(p.state(best)[2]-p.value(best))<1e-11
 result=dict(status=status,LB=lower,UB=ub,absolute_gap=ub-lower,relative_gap=(ub-lower)/ub,mask=np.where(best)[0].tolist(),expanded_nodes=expanded,pruned_nodes=pruned,unique_leaves=solved,frontier_nodes=len(heap),max_depth=max_depth,runtime=prior_runtime+time.monotonic()-start,cpu_seconds=prior_cpu+time.process_time()-cpu,root_LB=global_root_lb,pruned_floor=fathomed_floor,solver_bounds_used_for_pruning=False,best_updates=best_updates,all410_pages=True,full_query_count=64,numerical_tolerance=tol,certificate_scope='Original variable-denominator FP64 objective; full binary domain; node-valid convex supporting hyperplanes',node_journal_sha256=hashlib.sha256(record.read_bytes()).hexdigest())
 (root/'branch_frontier.json').write_text(json.dumps([dict(id=t[1],LB=t[0],fixed=t[2]) for t in heap],indent=2));(root/'branch_result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
