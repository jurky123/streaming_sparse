import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from .core import Problem
ROOT=Path('/tmp/e11_compact')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,required=True);a=ap.parse_args();root=ROOT/f'l{a.layer}';root.mkdir(parents=True,exist_ok=True);path=Path('/tmp/e11_results/contributions')/f'horse_field_s101_c5_l{a.layer}_s0.npz';data=dict(np.load(path));begin=time.monotonic();p=Problem(data);rng=np.random.default_rng(17)
 np.save(root/'gram.npy',p.G);np.savez_compressed(root/'compact.npz',e=p.e,d=p.d,od=p.od,C=p.C,own=p.own);checks=[]
 for name,x in [('mass',data['mass']),('risk',data['risk'])]+[(f'random{i}',np.isin(np.arange(410),np.r_[p.own,rng.choice(p.free,40,replace=False)])) for i in range(32)]:
  direct=p.value(x);gram=p.gram_value(x);res=p.state(x)[2];err=max(abs(direct-gram),abs(direct-res));assert err<max(1e-11,1e-8*abs(direct));checks.append(dict(name=name,direct=direct,gram=gram,residual=res,max_abs_error=err))
 # All 14,760 first-step exchanges checked, not a sampled candidate pool.
 for name in ('mass','risk'):
  _,_,g=p.swaps(data[name]);_,_,d=p.swaps(data[name],direct=True);err=float(np.max(np.abs(g-d)));assert err<max(1e-11,1e-8*p.value(data[name]));checks.append(dict(name=name+'_all14760_swaps',max_abs_error=err))
 (root/'validation.json').write_text(json.dumps(dict(passed=True,checks=checks,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),G_sha256=hashlib.sha256((root/'gram.npy').read_bytes()).hexdigest()),indent=2))
 runs=[];best=None;bv=float('inf')
 for i in range(8):
  if i<2:start=data[('mass','risk')[i]].copy();label=('mass','risk')[i]
  elif i<4:start=np.isin(np.arange(410),np.r_[p.own,rng.choice(p.free,40,replace=False)]);label=f'random{i-2}'
  else:
   start=best.copy();remove=rng.choice(np.where(start & (np.arange(410)!=p.own))[0],3,replace=False);add=rng.choice(np.where(~start)[0],3,replace=False);start[remove]=False;start[add]=True;label=f'perturb{i-4}'
  x,r=p.descend(start,label);runs.append(r)
  if r['final']<bv:best=x;bv=r['final']
  (root/'local_progress.json').write_text(json.dumps(dict(runs=runs,best=bv),indent=2));print('LOCAL',a.layer,label,r['initial'],r['final'],r['steps'],flush=True)
 two=[]
 for i in range(3):
  x,t=p.two_swap(best);two.append(t)
  if t['after']>=bv-max(1e-12,1e-9*abs(bv)):break
  best,r=p.descend(x,f'after_two_swap{i}');runs.append(r);bv=r['final']
 out=dict(layer=a.layer,P=410,Q=64,K=41,own=p.own,baseline_mass=p.value(data['mass']),baseline_risk=p.value(data['risk']),best_objective=bv,best_mask=np.where(best)[0].tolist(),runs=runs,two_swap=two,global_optimal=False,wall_seconds=time.monotonic()-begin)
 for name,x in [('mass',data['mass']),('risk',data['risk']),('incumbent',best)]:
  d=p.decomposition(x);np.savez_compressed(root/f'{name}_D02.npz',mask=x,**d)
 out['gain_over_mass']=1-bv/out['baseline_mass'];out['gain_over_risk']=1-bv/out['baseline_risk'];(root/'local_result.json').write_text(json.dumps(out,indent=2));print('FINISHED',a.layer,bv,out['gain_over_risk'],flush=True)
if __name__=='__main__':main()
