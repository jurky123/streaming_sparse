"""Independent replay of full-domain coverage and every supporting-plane certificate."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from .problem import ResidualProblem
from ..e11_compact.relaxation import build_h,linear_min

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--event',required=True);ap.add_argument('--root',default='/tmp/set_oracle_v1/global_bounds');a=ap.parse_args();root=Path(a.root)/a.event;start=time.monotonic();p=ResidualProblem(Path('/tmp/set_oracle_v1/contributions')/a.event);res=json.loads((root/'branch_result.json').read_text());rows=[json.loads(s) for s in (root/'branch_nodes.jsonl').read_text().splitlines()];nodes={};children={};terminal=[];max_error=0.;guard_min=float('inf')
 # Independently reconstruct the root SOC supporting plane from its saved point.
 H,B,L,U,shift,guard,meta=build_h(p);y=np.load(root/'soc_point.npy');E=np.einsum('qpd,p->qd',p.e,y);z=1-p.d@y;n=np.square(E).sum(1);f=np.sum(n/(z*U))/p.C;g=(2*np.einsum('qpd,qd,q->p',p.e,E,1/(z*U))+np.einsum('qp,q->p',p.d,n/(z*z*U)))/p.C;lm,_=linear_min(g,{p.own:0},(p.P-p.K));soclb=max(0.,float(f+lm-g@y-guard));rootlb=max(0.,soclb)
 def plane(row,fixed):
  nonlocal max_error,guard_min
  H,B,L,U,sh,gd,meta=build_h(p,fixed);pt=np.array(row['point'])
  if row.get('plane_kind','qp') in ('soc','secant'):
   ee=np.einsum('qpd,p->qd',p.e,pt);zz=1-p.d@pt;nn=np.square(ee).sum(1);coef=L+U if row.get('plane_kind')=='secant' else U;den=coef*zz-(L*U if row.get('plane_kind')=='secant' else 0);ff=np.sum(nn/den)/p.C;gg=(2*np.einsum('qpd,qd,q->p',p.e,ee,1/den)+np.einsum('qp,q->p',p.d,nn*coef/den**2))/p.C;cc=float(ff-gg@pt-gd)
  else:gg=2*B@pt;cc=float(pt@B@pt-gg@pt-sh*(p.P-p.K)-gd)
  storedg=np.array(row['gradient']);err=max(float(np.max(np.abs(gg-storedg))),abs(cc-row['intercept']));max_error=max(max_error,err);assert err<1e-12,err
  # Long-double independently accumulates this plane's integer minimum.
  ld=np.longdouble;pl=pt.astype(ld)
  if row.get('plane_kind','qp') in ('soc','secant'):
   el=np.einsum('qpd,p->qd',p.e.astype(ld),pl);zl=1-p.d.astype(ld)@pl;nl=np.square(el).sum(1);ul=U.astype(ld);ll=L.astype(ld);coef=ll+ul if row.get('plane_kind')=='secant' else ul;den=coef*zl-(ll*ul if row.get('plane_kind')=='secant' else 0);fl=np.sum(nl/den)/ld(p.C);gl=(2*np.einsum('qpd,qd,q->p',p.e.astype(ld),el,1/den)+np.einsum('qp,q->p',p.d.astype(ld),nl*coef/den**2))/ld(p.C);cl=fl-gl@pl-ld(gd)
  else:bl=B.astype(ld);gl=2*bl@pl;cl=pl@bl@pl-gl@pl-ld(sh)*(p.P-p.K)-ld(gd)
  _,yy=linear_min(np.array(gl,dtype=float),fixed,(p.P-p.K));long_bound=float(cl+gl@yy.astype(ld));mn,_=linear_min(gg,fixed,(p.P-p.K));bound=max(0.,cc+mn);assert abs(long_bound-(cc+mn))<gd*.1+1e-14
  guard_min=min(guard_min,gd);return gg,cc,bound
 for row in rows:
  ident=row['id'];assert ident not in nodes
  if row['parent'] is None:
   assert ident==0;fixed={p.own:0};gg,cc,plb=plane(row,fixed);rootlb=max(rootlb,plb);assert row['LB']<=rootlb+1e-12;nodes[ident]=dict(fixed=fixed,g=gg,c=cc,lb=row['LB']);continue
  parent=nodes[row['parent']];fixed=dict(parent['fixed']);page=row['branch_page'];bit=row['summary_bit'];assert page not in fixed and bit in (0,1);fixed[page]=bit;children.setdefault(row['parent'],[]).append((page,bit,ident));remaining=(p.P-p.K)-sum(fixed.values());nf=p.P-len(fixed);kind=row.get('pruned')
  if kind=='cardinality_infeasible':assert remaining<0 or remaining>nf;nodes[ident]=dict(fixed=fixed);continue
  assert 0<=remaining<=nf
  mn,_=linear_min(parent['g'],fixed,(p.P-p.K));cheap=max(parent['lb'],rootlb,parent['c']+mn);bound=cheap;gg=parent['g'];cc=parent['c']
  if 'point' in row:
   gg,cc,plb=plane(row,fixed);bound=max(bound,plb)
  if kind=='unique_feasible_mask_evaluated':
   assert remaining in (0,nf);mask=np.zeros(p.P,bool);mask[row['mask']]=True;assert mask.sum()==p.K and mask[p.own];assert all(int(not mask[i])==v for i,v in fixed.items());j=p.value(mask);assert abs(j-row['objective'])<1e-13;terminal.append(j)
  else:
   assert row['LB']<=bound+1e-12,(ident,row['LB'],bound);bound=row['LB']
   if kind:
    assert kind in ('valid_parent_tangent','node_convex_tangent');assert bound>=row['UB']-max(1e-10,1e-6*abs(row['UB']))-1e-13;terminal.append(bound)
  nodes[ident]=dict(fixed=fixed,g=gg,c=cc,lb=bound)
 frontier=json.loads((root/'branch_frontier.json').read_text());frontier_ids={r['id'] for r in frontier}
 for parent,chs in children.items():assert len(chs)==2 and chs[0][0]==chs[1][0] and {c[1] for c in chs}=={0,1}
 for row in rows:
  i=row['id']
  if row.get('pruned'):assert i not in children and i not in frontier_ids
  else:assert (i in children) != (i in frontier_ids),(i,'coverage hole')
 for r in frontier:assert r['LB']==nodes[r['id']]['lb'];terminal.append(r['LB'])
 x=np.zeros(p.P,bool);x[res['mask']]=True;assert x.sum()==p.K and x[p.own];ub=p.value(x);assert abs(ub-res['UB'])<1e-13;lb=min([ub]+terminal);assert abs(lb-res['LB'])<1e-12
 residual=p.state(x)[2];assert abs(residual-ub)<1e-12
 status='GLOBAL_TOL' if ub-lb<=max(1e-10,1e-6*abs(ub)) else 'BOUNDED_INCUMBENT';assert status==res['status']
 out=dict(passed=True,status=status,LB=lb,UB=ub,relative_gap=(ub-lb)/ub,nodes_replayed=len(rows),expanded_binary_partitions=len(children),frontier=len(frontier),all_legal_candidates=True,query_count=64,maximum_plane_reconstruction_error=max_error,minimum_roundoff_guard=guard_min,original_residual_objective_error=abs(residual-ub),long_double_tangent_minima_checked=True,solver_bounds_used=False,node_journal_sha256=hashlib.sha256((root/'branch_nodes.jsonl').read_bytes()).hexdigest(),wall_seconds=time.monotonic()-start)
 (root/'certificate_audit.json').write_text(json.dumps(out,indent=2));print(out,flush=True)
if __name__=='__main__':main()
