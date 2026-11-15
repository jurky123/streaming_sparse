from .coefficients import *
@torch.no_grad()
def audit_b():
 torch.set_num_threads(2);qs=torch.linspace(0,63,16,device='cuda').round().long();records=[];maxrel=0.;maxabs=0.
 for seed in [17,29]:
  dest=E27/f'B_s{seed}';done=json.loads((dest/'training_complete.json').read_text());assert done['frozen_C_exact'] and done['frozen_phi_exact'];assert done['phi_sha']==sha(E27/f's{seed}'/'selected.pt');model=Coefficients().cuda();model.load_state_dict(torch.load(dest/'selected.pt',weights_only=False,map_location='cuda')['model']);phi=load_phi(seed);b=bundle_for('C',seed);rr=json.loads((dest/f'development_epoch{done["best"]["epoch"]}.json').read_text());refs={(r['event'],r['method'],r['policy']):r for r in rr}
  for idx in [0,47,95,143,191,239,287]:
   path=files('development')[idx];d,s,c,tz,tb,lm=prepare(b,path);F=phi(d['q'],s);z,beta=model(s,F[qs],c['mass'][qs]);mask,r,a=route(d['q'][qs],s,b.layers[d['key']],d['own']);assert mask[d['own']];assert int(mask.sum())==max(1,int(len(mask)*.1));g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);normalizer=float(d['dense'][qs].square().sum()/16+1e-12)
   for name,be in [('PredictedBeta',beta),('TrueBeta',tb[qs])]:
    score,ind=safe_predictions(z,be,lm[qs],mask,r,a,normalizer)
    score_unscaled,_=safe_predictions(z,be,lm[qs],mask,r,a,1.);assert int(score.argmax())==int(score_unscaled.argmax()) and bool(score.max()>0)==bool(score_unscaled.max()>0)
    for policy,sc in [('Set',score),('Independent',ind)]:
     j=int(sc.argmax());ref=refs[(str(path.relative_to(OUT)),name,policy)];assert ref['remove']==int(r[j]) and ref['add']==int(a[j]) and ref['execute']==bool(sc[j]>0);assert abs(ref['executed_gain']-(float(g[j]) if sc[j]>0 else 0.))<1e-12
    zz=z.cpu().numpy().astype(np.longdouble);logs=lm[qs].cpu().numpy().astype(np.longdouble);be=be.cpu().numpy().astype(np.longdouble);mass=np.exp(logs);ex=np.exp(logs+be);u=ex-mass;ms=mask.cpu().numpy();Da=ex.sum(1);Ds=np.where(ms[None],ex,mass).sum(1);H=zz[:,ms].sum(1)/Ds[:,None];E=(zz[:,~ms].sum(1)-H*u[:,~ms].sum(1)[:,None])/Da[:,None]
    for j in [0,len(r)//2,int(score.argmax())]:
     rem,add=int(r[j]),int(a[j]);x=ms.copy();x[rem]=False;x[add]=True;den=np.where(x[None],ex,mass).sum(1);delta=(zz[:,add]-zz[:,rem]-H*(u[:,add]-u[:,rem])[:,None])/den[:,None];value=(2*(E*delta).sum(-1)-(delta**2).sum(-1)).mean()/normalizer;er=abs(float(value)-float(score[j]));rel=er/(1+abs(float(value)));maxrel=max(maxrel,rel);maxabs=max(maxabs,er);assert er<1e-8+1e-7*abs(float(value))
    records.append(dict(seed=seed,event=str(path.relative_to(OUT)),method=name))
 dump(E27/'B_independent_audit.json',dict(events=records,score_max_abs_error=maxabs,score_max_relative_error=maxrel,selected_replay_passed=True,inference_does_not_need_teacher_normalizer=True,frozen_C_and_phi_passed=True,test_accessed=False));print('B AUDIT PASS',maxrel)
if __name__=='__main__':audit_b()
