from .core import *
from ..e26_oracle_coefficients.report import stablecorr
from ..e25_decision_diagnostics.report import aggregate

def load_phi(seed,name='selected'):
 net=AdaptivePhi().cuda();net.load_state_dict(torch.load(E27/f's{seed}'/f'{name}.pt',map_location='cuda',weights_only=False)['model']);net.eval();return net

@torch.no_grad()
def analyze(seed):
 torch.set_num_threads(2);dest=E27/f's{seed}'
 assert (dest/'training_complete.json').exists()
 if (dest/'analysis_complete.json').exists():return
 bundle=bundle_for('C',seed);net=load_phi(seed);initial=load_phi(seed,'initial');qs=torch.linspace(0,63,16,device='cuda').round().long();fit=torch.load(OUT/'fit.pt',weights_only=True);rows=[];cond=[];errors=[];start=time.time();fitrows=[]
 for idx,path in enumerate(files('development')):
  d,s,c,tz,tb,lm=prepare(bundle,path);mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);p=np.load(E25/f'C_s{seed}'/f'panel_{idx:04d}.npz');event=str(path.relative_to(OUT));assert event==str(p['event']);assert np.array_equal(mask.cpu(),p['states'][0]);assert np.array_equal(r.cpu(),p['initial_remove']) and np.array_equal(a.cpu(),p['initial_add']);g=torch.as_tensor(p['true_initial_gain'],device='cuda');J=float(objective(c,d['N'],d['Z'],d['dense'],mask));norm=d['dense'][qs].square().sum()/16+1e-12;F=net(d['q'],s)
  Fi=initial(d['q'],s);Fs={'Adaptive64':F,'Adaptive16':F,'Untrained64':Fi}
  variants={'Adaptive64':ls_response(F,lm,tz,fit['zscale']),'Adaptive16':ls_response(F,lm,tz,fit['zscale'],qs),'Untrained64':ls_response(Fi,lm,tz,fit['zscale'])}
  out={};states=torch.as_tensor(p['states'],device='cuda');rr=torch.as_tensor(p['remove'],device='cuda');aa=torch.as_tensor(p['add'],device='cuda')
  for name,z in variants.items():
   X=((lm-lm.max(0).values[None]).exp()[...,None]*Fs[name]).permute(1,0,2);eigs=torch.linalg.eigvalsh(X.transpose(1,2)@X)
   fitrows.append(dict(event=event,seed=seed,method=name,phi_saturated_fraction=float((Fs[name][...,1:].abs()>.99).double().mean()),mean_design_rank=float((eigs>eigs[:,-1,None]*1e-10).sum(-1).double().mean()),z_nmse64=nmse(z,tz),z_nmse16=nmse(z[qs],tz[qs]),unselected_z_nmse64=nmse(z[:,~mask],tz[:,~mask])))
   score,ind=safe_predictions(z[qs],tb[qs],lm[qs],mask,r,a,norm);ps=[]
   for st in states:ps.append(safe_predictions(z[qs],tb[qs],lm[qs],st,rr,aa,norm)[0].cpu().numpy())
   ps=np.stack(ps);out[name]=ps;out['initial_'+name]=score.cpu().numpy();cond.append(dict(event=event,seed=seed,method=name,initial=J,**conditional_metrics(p['true_gain'],ps)))
   for policy,sc in [('Set',score),('Independent',ind)]:
    j=int(sc.argmax());ex=bool(sc[j]>0);true=float(g[j]);rows.append(dict(event=event,seed=seed,trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],method=name,policy=policy,initial=J,oracle_gain=max(0.,float(g.max())),proposal_gain=true,hindsight_gain=max(0.,true),executed_gain=true if ex else 0.,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(sc[j])))
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=np.array(event),states=p['states'],remove=p['remove'],add=p['add'],true_gain=p['true_gain'],**out)
  if idx%48==0:print('ANALYZE',seed,idx,flush=True)
 dump(dest/'decisions.json',rows);dump(dest/'conditional.json',cond);dump(dest/'fit_quality.json',fitrows);dump(dest/'analysis_complete.json',dict(seed=seed,events=288,seconds=time.time()-start,selected_sha=sha(dest/'selected.pt'),C_sha=sha(checkpoint('C',seed)),test_accessed=False));print('ANALYZED',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args();analyze(args.seed)
