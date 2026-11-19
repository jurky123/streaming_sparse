"""Frozen-output coverage and head9 gradients before any T1 update."""
from .core import *
from scipy.special import softmax

def run(seed):
 torch.set_num_threads(2);protocol();data=Data(seed);contract(seed,data);meta=json.loads((CACHE/f'train_s{seed}/metadata.json').read_text());rows=[];checks=[];qs=np.rint(np.linspace(0,63,16)).astype(int)
 for mm in meta:
  x=np.load(CACHE/f'train_s{seed}'/('event_%04d.npz'%mm['index']));lm=x['logmass'][qs];m=np.exp(lm);alpha=softmax(lm+x['beta'][qs],axis=1);w0=m*m;w1=.5*w0+.5*w0.sum(1,keepdims=True)*alpha**2/(alpha**2).sum(1,keepdims=True);err=x['local_mse'][qs];under=m<alpha/10
  ce=float(np.max(abs(w1.sum(1)-w0.sum(1))));assert ce<1e-12 and np.all(w1>=.5*w0)
  flags={role:np.broadcast_to(x['role_'+role],m.shape) for role in ['all','harmful_predicted_endpoint','oracle_endpoint','endpoint_pair','background']}
  base={k:mm[k] for k in ['seed','event','trajectory','layer','head','chunk','step','tile']}
  for role,reg in flags.items():
   for status,sel in [('all',reg),('under10',reg&under),('other',reg&~under)]:
    if not sel.any():continue
    for arm,w in [('T0',w0),('T1',w1)]:
     loss=w*err;grad=w*np.sqrt(16*err)
     rows.append(dict(**base,role=role,status=status,arm=arm,query_pages=int(sel.sum()),true_mass_share=float(alpha[sel].sum()/16),weight_share=float(w[sel].sum()/w.sum()),loss_share=float(loss[sel].sum()/loss.sum()),gradient_l2_share=float(grad[sel].sum()/grad.sum()),loss_sum=float(loss.sum()/m.size/data.scale**2),weight_top1_share=float((w.max(1)/w.sum(1)).mean()),weight_top5_share=float((np.sort(w,axis=1)[:,-5:].sum(1)/w.sum(1)).mean())))
  checks.append(dict(event=mm['event'],weight_conservation_error=ce,lower_bound_ok=True))
 table(D/f'coverage_s{seed}.csv',rows);table(D/f'weight_checks_s{seed}.csv',checks)
 selected=json.loads((D/'protocol.json').read_text())['head9_events'][str(seed)];selected=list(dict.fromkeys(selected+json.loads((D/'protocol.json').read_text())['worst_events']));gradrows=[];gc=[];model=model_from(E29/f's{seed}/selected.pt');frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
 for event in selected:
  d=data.load(event);base={k:d['meta'][k] for k in ['seed','event','trajectory','layer','head','chunk','step','tile']};assert base['head']==9
  for arm,new in [('T0',False),('T1',True)]:
   ls,rp,bp,_=loss_terms(model,d,data,new);gradrows+=parameter_audit(model,ls,dict(**base,arm=arm));actual=torch.autograd.grad(ls[0],rp)[0];w=d['w1'] if new else d['w0'];expected=2*w[...,None]*(rp-d['rtrue'])/data.scale**2/rp.numel();assert torch.allclose(actual.double(),expected,rtol=3e-5,atol=1e-9)
   gc.append(dict(**base,arm=arm,local_loss=float(ls[0]),local_gradient_max_error=float((actual-expected).abs().max()),finite=bool(torch.isfinite(actual).all())))
 assert all(torch.equal(v,frozen[k]) for k,v in model.state_dict().items());data.check_frozen();table(D/f'head9_gradients_s{seed}.csv',gradrows);table(D/f'head9_checks_s{seed}.csv',gc)
 dump(D/f'preflight_s{seed}.json',dict(seed=seed,events=1152,head9_events=selected,weights_unchanged=True,max_weight_error=max(r['weight_conservation_error'] for r in checks),finite=True));print('PREFLIGHT DONE',seed,flush=True)

def gate():
 rows=[];summ=[]
 for seed in [17,29]:
  assert (D/f'preflight_s{seed}.json').exists()
  with (D/f'coverage_s{seed}.csv').open() as f:rs=list(csv.DictReader(f))
  gs=collections.defaultdict(list)
  for r in rs:gs[(r['role'],r['status'],r['arm'])].append(r)
  for (role,status,arm),v in gs.items():
   row=dict(seed=seed,role=role,status=status,arm=arm,events=len(v))
   for k in ['true_mass_share','weight_share','loss_share','gradient_l2_share','loss_sum','weight_top1_share','weight_top5_share']:row[k]=float(np.mean([float(r[k]) for r in v]))
   summ.append(row)
  for role in ['all','harmful_predicted_endpoint','oracle_endpoint']:
   old=next(r for r in summ if r['seed']==seed and r['role']==role and r['status']=='under10' and r['arm']=='T0');new=next(r for r in summ if r['seed']==seed and r['role']==role and r['status']=='under10' and r['arm']=='T1');passed=all(new[k]>old[k] for k in ['weight_share','loss_share','gradient_l2_share']);rows.append(dict(seed=seed,role=role,passed=passed,old=old,new=new))
 table(D/'coverage_summary.csv',summ);dump(D/'preflight_gate.json',dict(passed=all(r['passed'] for r in rows),criteria=rows,no_weight_search=True));print('GATE',all(r['passed'] for r in rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--gate',action='store_true');a=p.parse_args();gate() if a.gate else run(a.seed)
