from .analyze import *
@torch.no_grad()
def audit():
 torch.set_num_threads(2);fit=torch.load(OUT/'fit.pt',weights_only=True);qs=torch.linspace(0,63,16,device='cuda').round().long();records=[];independent_checks=[];maxfiterr=0.;maxscoreerr=0.;maxrelerr=0.
 for seed in [17,29]:
  net=load_phi(seed);bundle=bundle_for('C',seed);train_done=json.loads((E27/f's{seed}'/'training_complete.json').read_text());assert train_done['frozen_C_exact'] and train_done['C_sha']==sha(checkpoint('C',seed))
  for idx in [0,47,95,143,191,239,287]:
   d,s,c,tz,tb,lm=prepare(bundle,files('development')[idx]);F=net(d['q'],s);z=ls_response(F,lm,tz,fit['zscale']);xc=((lm-lm.max(0).values[None]).exp()[...,None]*F).permute(1,0,2)
   for page in [0,len(xc)//2,len(xc)-1]:
    x=xc[page].cpu().numpy();y=tz[:,page].cpu().numpy();scale=max(np.max(abs(y)),1e-300);reg=RIDGE*np.sum(x*x)/5;aug=np.concatenate([x,np.sqrt(reg)*np.eye(5)]);rhs=np.concatenate([y/scale,np.zeros((5,16))]);coef=np.linalg.lstsq(aug,rhs,rcond=None)[0];er=float(np.max(abs(x@coef-z[:,page].cpu().numpy()/scale)));maxfiterr=max(maxfiterr,er);assert er<1e-7
   mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);assert int(mask.sum())==max(1,int(len(mask)*.1)) and mask[d['own']];norm=float(d['dense'][qs].square().sum()/16+1e-12);score,independent= safe_predictions(z[qs],tb[qs],lm[qs],mask,r,a,norm)
   # long-double reference, same fitted coordinates but no common-exp rounding.
   zz=z[qs].cpu().numpy().astype(np.longdouble);logs=lm[qs].cpu().numpy().astype(np.longdouble);be=tb[qs].cpu().numpy().astype(np.longdouble);mass=np.exp(logs);ex=np.exp(logs+be);tar=zz.sum(1)/ex.sum(1)[:,None];ms=mask.cpu().numpy()
   def obj(mm):
    den=np.where(mm[None],ex,mass).sum(1);o=zz[:,mm].sum(1)/den[:,None];return ((tar-o)**2).sum(-1).mean()/norm
   indnet=zz-zz.sum(1)[:,None]/ex.sum(1)[:,None,None]*(ex-mass)[...,None];energy=((indnet/ex.sum(1)[:,None,None])**2).sum(-1).mean(0);ind=energy[a.cpu().numpy()]-energy[r.cpu().numpy()];ji=int(ind.argmax());jr=int(independent.argmax());same=ji==jr or abs(ind[ji]-ind[jr])<=1e-10*max(1,abs(ind[ji]));assert same and bool(ind[ji]>0)==bool(independent.max()>0);independent_checks.append(dict(seed=seed,event=str(files('development')[idx].relative_to(OUT)),same_action_or_tie=bool(same),same_noop=True))
   base=obj(ms)
   # Long-double incremental identity avoids cancellation even in the reference.
   Da=ex.sum(1);Ds=np.where(ms[None],ex,mass).sum(1);H=zz[:,ms].sum(1)/Ds[:,None];u=ex-mass;E=(zz[:,~ms].sum(1)-H*u[:,~ms].sum(1)[:,None])/Da[:,None]
   for j in [0,len(r)//2,int(score.argmax())]:
    mm=ms.copy();mm[int(r[j])]=False;mm[int(a[j])]=True;den=np.where(mm[None],ex,mass).sum(1);delta=(zz[:,int(a[j])]-zz[:,int(r[j])]-H*(u[:,int(a[j])]-u[:,int(r[j])])[:,None])/den[:,None];val=(2*(E*delta).sum(-1)-(delta**2).sum(-1)).mean()/norm;er=abs(float(val)-float(score[j]));rel=er/(1+abs(float(val)));maxscoreerr=max(maxscoreerr,er);maxrelerr=max(maxrelerr,rel);assert er<1e-8+1e-7*abs(float(val)), dict(seed=seed,idx=idx,action=j,reference=float(val),score=float(score[j]),error=er)
   records.append(dict(seed=seed,event=str(files('development')[idx].relative_to(OUT))))
 dump(E27/'independent_audit.json',dict(augmented_numpy_fit_max_scaled_error=maxfiterr,longdouble_score_max_abs_error=maxscoreerr,longdouble_score_max_relative=maxrelerr,events=records,score_tolerance='1e-8+1e-7*abs(reference)',independent_checks=independent_checks,frozen_C_checks_passed=True,test_accessed=False));print('AUDIT PASS',maxfiterr,maxrelerr)
if __name__=='__main__':audit()
