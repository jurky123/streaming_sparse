"""Signed-log I/B diagnostics for extreme teacher-beta fits; no clipping."""
from .run import *
import csv

def signed_components(rp,beta,lm,center,mask,r,a,norm):
 le=lm+beta;shift=torch.maximum(lm.max(1).values,le.max(1).values)
 m=(lm-shift[:,None]).exp();ex=(le-shift[:,None]).exp();v=rp+center;target=(le.softmax(1)[...,None]*v).sum(1)
 net=ex[...,None]*(v-target[:,None])-m[...,None]*(center-target[:,None]);E=net[:,~mask].sum(1);logDs=torch.where(mask[None],le,lm).logsumexp(1)
 f=2*(shift-logDs)-math.log(len(lm))-math.log(float(norm));Is=[];Il=[];Bl=[]
 for i in range(0,len(r),32):
  delta=net[:,r[i:i+32]]-net[:,a[i:i+32]];iq=-2*(E[:,None]*delta).sum(-1);bq=delta.square().sum(-1)
  lp=torch.where(iq>0,iq.abs().log()+f[:,None],-torch.inf).logsumexp(0);ln=torch.where(iq<0,iq.abs().log()+f[:,None],-torch.inf).logsumexp(0)
  sign=torch.sign(lp-ln);sign=torch.where(lp==ln,torch.zeros_like(sign),sign)
  logabs=torch.maximum(lp,ln)+torch.log(-torch.expm1(-abs(lp-ln)));logabs=torch.where(lp==ln,torch.full_like(logabs,-torch.inf),logabs)
  Is.append(sign);Il.append(logabs);Bl.append((bq.log()+f[:,None]).logsumexp(0))
 return torch.cat(Is),torch.cat(Il),torch.cat(Bl)

@torch.no_grad()
def main(seed):
 torch.set_num_threads(2);rows=[];dest=DEST/f's{seed}';information=json.loads((dest/'information.json').read_text());oldinfo={(r['event'],r['method'],r['queries'],r['region']):r for r in information};qs=torch.linspace(0,63,16).round().long();paths=files('development')
 for i in range(288):
  x=torch.load(dest/f'beta_{i:04d}.pt',weights_only=True);p=np.load(dest/f'panel_{i:04d}.npz');ev=torch.load(paths[i],weights_only=True);norm=ev['dense'][qs].double().square().sum()/16+1e-12;mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');rp=x['predicted_response'][qs].cuda().double();lm=x['logmass'][qs].cuda();c=x['center'][qs].cuda();tb=torch.as_tensor(p['trueB'],device='cuda');ti=torch.as_tensor(p['trueI'],device='cuda');out={}
  for method in METHODS:
   for qname,qq in [('M16',qs),('Q64',torch.arange(64))]:
    cand=torch.zeros(x['logmass'].shape[1],dtype=torch.bool);cand[p['initial_remove']]=True;cand[p['initial_add']]=True
    for region,select in [('all',torch.ones_like(cand)),('candidate',cand),('background',~cand)]:
     oldinfo[(x['event'],method,qname,region)].update(weighted_metrics(x['betas'][method][qq],x['beta_true'][qq],x['logmass'][qq],select))
   si,li,lb=signed_components(rp,x['betas'][method][qs].cuda().double(),lm,c,mask,r,a,norm)
   validB=(tb>0)&torch.isfinite(lb);err=(lb[validB]-tb[validB].log()).abs()
   rows.append(dict(seed=seed,event=x['event'],method=method,I_sign_accuracy=float(((si>0)==(ti>0)).double().mean()),I_zero_count=int((si==0).sum()),B_log_abs_error_mean=float(err.mean()) if len(err) else None,B_log_valid_actions=int(validB.sum()),I_unrepresentable_fp64=int((li>math.log(np.finfo(float).max)).sum()),B_unrepresentable_fp64=int((lb>math.log(np.finfo(float).max)).sum()),all_signed_logs_valid=bool((~torch.isnan(li)).all() and (~torch.isnan(lb)).all())))
   assert rows[-1]['all_signed_logs_valid'];out[method+'_I_sign']=si.cpu().numpy();out[method+'_I_logabs']=li.cpu().numpy();out[method+'_B_log']=lb.cpu().numpy()
  np.savez_compressed(dest/f'components_{i:04d}.npz',**out)

 for rr in information:
  for k,v in rr.items():
   if isinstance(v,float) and not math.isfinite(v):rr[k]=None
 dump(dest/'information.json',information);dump(dest/'signed_components.json',rows);print('SIGNED COMPONENTS',seed,len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.seed)
