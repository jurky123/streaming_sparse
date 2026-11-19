from .core import *
from .report import table

def arr(x):return x.detach().cpu().double().numpy().astype(np.longdouble)
def sm(x):
 w=np.exp(x-x.max(1,keepdims=True));return w/w.sum(1,keepdims=True)
def numpy_score(rp,bp,lm,cen,mask,rs,aa,norm):
 v=rp+cen;D=(sm(lm+bp)[...,None]*v).sum(1)
 def h(m):return (sm(np.where(m[None],lm+bp,lm))[...,None]*(cen+m[None,:,None]*rp)).sum(1)
 H=h(mask);out=[]
 for r,a in zip(rs,aa):
  new=mask.copy();new[r]=False;new[a]=True;delta=h(new)-H;out.append(float((2*(D-H)*delta-delta*delta).sum(-1).mean()/norm))
 return out

def main():
 torch.set_num_threads(2);qs=np.linspace(0,63,16).round().astype(int);checks=[];derivatives=[];closed=[];hashes={};weightrows=[];rolecounts=[];replay=[]
 for seed in [17,29]:
  bd=DEST/f'B_s{seed}';ad=DEST/f'A_s{seed}';assert (bd/'complete.json').exists() and (ad/'complete.json').exists();br=json.loads((bd/'action_margin_attribution.json').read_text());reg=json.loads((bd/'region_attribution.json').read_text());ar=json.loads((ad/'decisions.json').read_text());byevent=collections.defaultdict(list)
  for r in ar:byevent[r['event']].append(r)
  for idx,path in enumerate(files('development')):
   xfile=E30/f's{seed}/beta_{idx:04d}.pt';pf=E30/f's{seed}/panel_{idx:04d}.npz';tfile=bd/f'target_{idx:04d}.pt';x=torch.load(xfile,map_location='cpu',weights_only=True);p=np.load(pf);ap=np.load(ad/f'panel_{idx:04d}.npz');t=torch.load(tfile,weights_only=True);event=t['event'];assert event==x['event']==str(p['event'])==str(ap['event']);meta=t['meta'];norm=t['norm64'];mask=p['states'][0];rs=p['initial_remove'];aa=p['initial_add'];C=np.zeros(len(mask),bool);C[rs]=True;C[aa]=True
   for k in ['states','remove','add','initial_remove','initial_add','true_gain','true_initial_gain']:assert np.array_equal(ap[k],p[k])
   z=br[idx];rr=reg[idx];bpanel=np.load(bd/f'panel_{idx:04d}.npz');assert abs(z['phi_response']+z['phi_beta']-(z['M00']-z['M11']))<1e-9;assert abs(rr['phi_endpoint']+rr['phi_other']+rr['phi_background']-(rr['M000']-rr['M111']))<1e-9;assert rr['end_pages']+rr['other_candidate_pages']==48;assert z['same_action']==(z['j_pred']==z['j_oracle']);closed.append(dict(seed=seed,event=event,branch_error=abs(z['phi_response']+z['phi_beta']-z['M00']+z['M11']),region_error=abs(rr['phi_endpoint']+rr['phi_other']+rr['phi_background']-rr['M000']+rr['M111'])))
   for row in byevent[event]:
    j=row['action'];new=mask.copy()
    if j>=0:
     r,a=rs[j],aa[j];assert mask[r] and not mask[a] and r!=len(mask)-72+meta['tile'];new[r]=False;new[a]=True;assert row['executed_gain']==float(p['true_initial_gain'][j])
    else:assert row['executed_gain']==0.
    assert new.sum()==int(.1*len(mask)) and new[len(mask)-72+meta['tile']]
   # Positive offline scale replay: selections, not equality of differently normalized scores.
   for method,key in [('E29','initial_Predicted'),('TrueBeta','initial_True')]:
    assert pick(torch.from_numpy(ap['initial_'+method]))==pick(torch.from_numpy(p[key]));old=np.load(ROOT/f'artifacts/e37_shared_beta_v1/s{seed}/panel_{idx:04d}.npz');m='E29' if method=='E29' else 'TrueBeta';assert np.array_equal(np.argsort(-ap['initial_'+method]),np.argsort(-old['initial_'+m])) or np.allclose(ap['initial_'+method]/max(1e-300,abs(ap['initial_'+method]).max()),old['initial_'+m]/max(1e-300,abs(old['initial_'+m]).max()),atol=1e-8)
   old=np.load(ROOT/f'artifacts/e37_shared_beta_v1/s{seed}/panel_{idx:04d}.npz');assert pick(torch.from_numpy(ap['initial_G2-mass']))==pick(torch.from_numpy(old['initial_CandidateBackground-Q']));replay.append(dict(seed=seed,event=event,baseline_truebeta_two_group_replay=True))
   # Weight mass by layer, and epsilon share, computed independently with NumPy for initial state.
   lm=arr(x['logmass'])[qs];rp=arr(x['predicted_response'])[qs];bp=arr(x['betas']['Predicted'])[qs];bt=arr(x['beta_true'])[qs];cen=arr(x['center'])[qs];rt=arr(t['true_response'])[qs];v=cen+rp;alpha=sm(lm+bp);omega=sm(np.where(mask[None],lm+bp,lm));D=(alpha[...,None]*v).sum(1);H=(omega[...,None]*(cen+mask[None,:,None]*rp)).sum(1);jac=alpha[...,None]*(v-D[:,None])-mask[None,:,None]*omega[...,None]*(v-H[:,None]);j2=(jac*jac).sum(-1);dir2=(v*v).sum(-1);logweights={'mass':2*lm,'direction':2*lm+np.log(dir2+EPS),'jacobian':np.log(j2+EPS)}
   def lse(a):
    mx=a.max();return float(mx+np.log(np.exp(a-mx).sum()))
   for kind,lw in logweights.items():weightrows.append(dict(**meta,weight=kind,log_total_weight=lse(lw),epsilon_fraction=float(np.mean(EPS/(j2+EPS))) if kind=='jacobian' else float(np.mean(EPS/(dir2+EPS))) if kind=='direction' else 0.))
   for f in [xfile,pf,tfile,path]:hashes[str(f.relative_to(ROOT))]=sha(f)
   if idx not in [0,47,143,144,230,287]:continue
   # Independent direct-output margins, all four branch and eight region vertices.
   pair=[z['j_pred'],z['j_oracle']]
   def margin_np(rp_,bp_):
    jj=[j for j in pair if j>=0];val=iter(numpy_score(rp_,bp_,lm,cen,mask,rs[jj],aa[jj],norm));vals=[next(val) if j>=0 else 0. for j in pair];return vals[0]-vals[1]
   for name,rp_,bp_ in [('M00',rp,bp),('M10',rt,bp),('M01',rp,bt),('M11',rt,bt)]:
    error=abs(margin_np(rp_,bp_)-z[name]);assert error<1e-8+1e-6*abs(z[name]);checks.append(dict(seed=seed,event=event,check=name,error=error))
   ep=np.zeros(len(mask),bool)
   for j in pair:
    if j>=0:ep[[rs[j],aa[j]]]=True
   regions=[ep,C&~ep,~C]
   for bits in range(8):
    use=np.zeros(len(mask),bool)
    for k in range(3):
     if bits&(1<<k):use|=regions[k]
    got=margin_np(np.where(use[None,:,None],rt,rp),np.where(use[None],bt,bp));err=abs(got-rr[f'M{bits:03b}']);assert err<1e-8+1e-6*abs(got);checks.append(dict(seed=seed,event=event,check='region'+str(bits),error=err))
   for ng in [2,4]:
    groups=[C,~C] if ng==2 else [C&mask,C&~mask,~C&mask,~C&~mask];assert np.stack(groups).sum(0).min()==1 and np.stack(groups).sum(0).max()==1
    for kind,lw in logweights.items():
     beta=bp.copy()
     for ix in groups:
      if not ix.any():continue
      w=sm(lw[:,ix]);b=(w*(bt[:,ix]-bp[:,ix])).sum(1);beta[:,ix]+=b[:,None]
     method=f'G{ng}-{kind}';row=next(r for r in byevent[event] if r['method']==method and r['policy']=='Set');j=int(np.argmax(ap['initial_'+method]));got=numpy_score(rp,beta,lm,cen,mask,rs[[j]],aa[[j]],norm)[0];err=abs(got-ap['initial_'+method][j]);assert err<1e-8+1e-6*abs(got);checks.append(dict(seed=seed,event=event,check=method,error=err))
   if idx in [0,143,230]:
    q=0;br0=x['betas']['Predicted'][qs[q]].double().cuda().requires_grad_(True);r0=x['predicted_response'][qs[q]].double().cuda();l0=x['logmass'][qs[q]].cuda();c0=x['center'][qs[q]].cuda();m=torch.as_tensor(mask,device='cuda')
    def errorfun(b):
     D,H,_,_=mix(r0[None],b[None],l0[None],c0[None],m);return (D-H)[0]
    actual=torch.autograd.functional.jacobian(errorfun,br0).T;delta=float(np.max(np.abs(arr(actual)-jac[q])));assert delta<1e-10;derivatives.append(dict(seed=seed,event=event,max_jacobian_error=delta))
  done=json.loads((bd/'complete.json').read_text());assert done['protocol_sha']==sha(DEST/'protocol.json')
  for f,key in [(checkpoint('C',seed),'C_sha'),(E29/f's{seed}/selected.pt','reader_sha'),(E27/f's{seed}/selected.pt','phi_sha'),(OUT/'fit.pt','basis_sha')]:assert sha(f)==done[key];hashes[str(f.relative_to(ROOT))]=sha(f)
 for f in (ROOT/'local_experiments/e38_action_attribution').glob('*.py'):hashes[str(f.relative_to(ROOT))]=sha(f)
 table('source_hashes.csv',[dict(path=k,sha256=v) for k,v in sorted(hashes.items())]);table('raw_weight_diagnostics.csv',weightrows)
 shares=[]
 for seed in [17,29]:
  for kind in ['mass','direction','jacobian']:
   ww=[r for r in weightrows if r['seed']==seed and r['weight']==kind];mx=max(r['log_total_weight'] for r in ww);den=sum(math.exp(r['log_total_weight']-mx) for r in ww)
   for layer in [1,15,29]:
    sub=[r for r in ww if r['layer']==layer];shares.append(dict(seed=seed,weight=kind,layer=layer,raw_weight_fraction=sum(math.exp(r['log_total_weight']-mx) for r in sub)/den,mean_epsilon_fraction=float(np.mean([r['epsilon_fraction'] for r in sub]))))
 table('weight_layer_shares.csv',shares);dump(DEST/'audit.json',dict(events=576,branch_region_closure=closed,independent_longdouble=checks,jacobian_autograd=derivatives,baseline_replay=replay,all_frozen_hashes_unchanged=True,test_accessed=False,new_capture=False,existing_KV_reconstruction_authorized=True));print('AUDIT PASSED',len(closed),len(checks),len(derivatives),flush=True)
if __name__=='__main__':main()
