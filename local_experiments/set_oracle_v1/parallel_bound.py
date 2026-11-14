"""Resume a verified full-domain frontier; CPU workers evaluate child bounds, coordinator commits proof records."""
import argparse,json,time,heapq,hashlib,shutil,multiprocessing,concurrent.futures
from pathlib import Path
import numpy as np
from .problem import ResidualProblem
from ..e11_compact.relaxation import build_h,qp_relax,soc_projected,linear_min
P=None

def child_bound(job):
 start=time.process_time();serial,parent,var,bit,ub,rootlb=job;lb,ident,fixed,point,grad,intercept=parent;child=dict(fixed);child[var]=bit;remaining=(P.P-P.K)-sum(child.values());free=np.array([i for i in range(P.P) if i not in child]);rec=dict(id=serial,parent=ident,branch_page=var,summary_bit=bit,depth=len(child)-1)
 def out(**kw):return dict(rec=rec,fixed=child,cpu=time.process_time()-start,**kw)
 if remaining<0 or remaining>len(free):rec['pruned']='cardinality_infeasible';return out(kind='infeasible')
 linear,_=linear_min(grad,child,(P.P-P.K));cheap=max(lb,rootlb,intercept+linear)
 if cheap>=ub-max(1e-10,1e-6*ub):rec.update(pruned='valid_parent_tangent',LB=cheap,UB=ub);return out(kind='pruned',bound=cheap)
 if remaining in (0,len(free)):
  yy=np.zeros(P.P,bool)
  for i,v in child.items():yy[i]=v
  if remaining:yy[free]=True
  x=~yy;j=P.value(x);rec.update(pruned='unique_feasible_mask_evaluated',objective=j,mask=np.where(x)[0].tolist());return out(kind='leaf',bound=j,candidate=x,objective=j)
 H,B,L,U,sh,guard,meta=build_h(P,child);q=qp_relax(P,H,B,sh,guard,child,initial=point,maxiter=350);soc=soc_projected(P,U,guard,child,q['point'],maxiter=100,L=L);use=soc['LB']>q['LB'];q=soc if use else q;bound=max(cheap,q['LB']);pt=q['point'];g=q['gradient'] if use else 2*B@pt;c=q['intercept'] if use else float(pt@B@pt-g@pt-sh*(P.P-P.K)-guard)
 yy=np.zeros(P.P,bool)
 for i,v in child.items():yy[i]=v
 if remaining:yy[free[np.argsort(pt[free])[-remaining:]]]=True
 x=~yy;j=P.value(x);rec.update(LB=bound,denominator_U_min=float(U.min()),denominator_U_max=float(U.max()),roundoff_guard=guard,plane_kind='secant' if use else 'qp',fixed=child,point=pt.tolist(),gradient=g.tolist(),intercept=c)
 return out(kind='node',bound=bound,point=pt,gradient=g,intercept=c,candidate=x,objective=j)

def main():
 global P
 ap=argparse.ArgumentParser();ap.add_argument('--event',required=True);ap.add_argument('--workers',type=int,default=16);ap.add_argument('--seconds',type=float,default=150);a=ap.parse_args();source=Path('/tmp/set_oracle_v1/global_bounds')/a.event;root=Path('/tmp/set_oracle_v1/global_bounds_parallel')/a.event;root.mkdir(parents=True,exist_ok=True)
 for name in ('branch_nodes.jsonl','branch_frontier.json','branch_result.json','soc_point.npy','soc_relaxation.json'):shutil.copy2(source/name,root/name)
 previous=json.loads((root/'branch_result.json').read_text());records={r['id']:r for r in map(json.loads,(root/'branch_nodes.jsonl').read_text().splitlines())};front=json.loads((root/'branch_frontier.json').read_text());heap=[]
 for r in front:
  rr=records[r['id']];heap.append((r['LB'],r['id'],{int(k):v for k,v in r['fixed'].items()},np.array(rr['point']),np.array(rr['gradient']),rr['intercept']))
 heapq.heapify(heap);P=ResidualProblem(Path('/tmp/set_oracle_v1/contributions')/a.event);build_h(P);P.Eflat=np.ascontiguousarray(P.e.transpose(0,2,1).reshape(P.Q*P.D,P.P));best=np.zeros(P.P,bool);best[previous['mask']]=True;ub=P.value(best);rootlb=previous['root_LB'];floor=previous['pruned_floor'];serial=max(records);expanded=previous['expanded_nodes'];pruned=previous['pruned_nodes'];solved=previous['unique_leaves'];depth=previous['max_depth'];updates=previous['best_updates'];start=time.monotonic();cpu=time.process_time();worker_cpu=0.;batches=0;path=root/'branch_nodes.jsonl'
 with path.open('a') as journal,concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
  while heap and time.monotonic()-start<a.seconds and heap[0][0]<ub-max(1e-10,1e-6*ub):
   jobs=[]
   for _ in range(min(len(heap),max(1,a.workers//2))):
    parent=heapq.heappop(heap);lb,ident,fixed,point,grad,intercept=parent;free=np.array([i for i in range(P.P) if i not in fixed]);frac=np.minimum(point[free],1-point[free]);uncert=np.std(P.d[:,free],axis=0);var=int(free[np.argmax(frac+1e-5*uncert/max(float(uncert.max()),1e-20))]);expanded+=1
    for bit in (0,1):serial+=1;jobs.append((serial,parent,var,bit,ub,rootlb))
   # Complete the entire batch before checking the clock: no missing children at timeout.
   for got in pool.map(child_bound,jobs):
    rec=got['rec'];worker_cpu+=got['cpu'];depth=max(depth,rec['depth']);kind=got['kind']
    if 'objective' in got and got['objective']<ub:best=got['candidate'];ub=got['objective'];updates.append(dict(node=rec['id'],objective=ub,mask=np.where(best)[0].tolist()))
    if kind=='infeasible':pruned+=1
    elif kind in ('pruned','leaf'):
     floor=min(floor,got['bound']);pruned+=int(kind=='pruned');solved+=int(kind=='leaf')
    else:
     rec['UB']=ub
     if got['bound']>=ub-max(1e-10,1e-6*ub):rec['pruned']='node_convex_tangent';pruned+=1;floor=min(floor,got['bound'])
     else:heapq.heappush(heap,(got['bound'],rec['id'],got['fixed'],got['point'],got['gradient'],got['intercept']))
    journal.write(json.dumps(rec)+'\n')
   batches+=1;journal.flush();lower=min(ub,floor,heap[0][0] if heap else float('inf'));progress=dict(event_id=a.event,workers=a.workers,batches=batches,expanded=expanded,frontier=len(heap),LB=lower,UB=ub,relative_gap=(ub-lower)/ub,seconds=time.monotonic()-start,worker_cpu_seconds=worker_cpu);(root/'branch_progress.json').write_text(json.dumps(progress,indent=2));print(progress,flush=True)
 lower=min(ub,floor,heap[0][0] if heap else float('inf'));tol=max(1e-10,1e-6*ub);status='GLOBAL_TOL' if ub-lower<=tol else 'BOUNDED_INCUMBENT';assert best.sum()==P.K and best[P.own] and abs(P.value(best)-P.state(best)[2])<1e-11
 result=dict(previous,status=status,LB=lower,UB=ub,absolute_gap=ub-lower,relative_gap=(ub-lower)/ub,mask=np.where(best)[0].tolist(),expanded_nodes=expanded,pruned_nodes=pruned,unique_leaves=solved,frontier_nodes=len(heap),max_depth=depth,pruned_floor=floor,best_updates=updates,runtime=previous['runtime']+time.monotonic()-start,cpu_seconds=previous['cpu_seconds']+worker_cpu+time.process_time()-cpu,parallel_workers=a.workers,additional_wall_seconds=time.monotonic()-start,additional_worker_cpu_seconds=worker_cpu,checkpoint_source=str(source),node_journal_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 (root/'branch_frontier.json').write_text(json.dumps([dict(id=t[1],LB=t[0],fixed=t[2]) for t in heap],indent=2));(root/'branch_result.json').write_text(json.dumps(result,indent=2));print('FINISHED',status,lower,ub,flush=True)
if __name__=='__main__':main()
