from .core import *
@torch.no_grad()
def audit():
 torch.set_num_threads(2);torch.manual_seed(1701);device='cuda';Q,P,D=64,12,16
 F=torch.cat([torch.ones(Q,P,1,device=device,dtype=torch.double),torch.randn(Q,P,4,device=device,dtype=torch.double)],-1)
 logmass=torch.randn(Q,P,device=device,dtype=torch.double).log_softmax(1);target=torch.randn(Q,P,D,device=device,dtype=torch.double);beta=torch.randn(Q,P,device=device,dtype=torch.double)
 z,b,co,info=fit_coeff(F,logmass,target,beta,.053,1.53);X=(logmass.exp()[...,None]*F).permute(1,0,2);G=X.transpose(1,2)@X;reg=RIDGE*G.diagonal(dim1=-2,dim2=-1).sum(-1)/5;C=torch.linalg.solve(G+reg[:,None,None]*torch.eye(5,device=device,dtype=torch.double),X.transpose(1,2)@target.permute(1,0,2));direct=(X@C).permute(1,0,2);ferr=float((z-direct).abs().max());assert ferr<1e-10
 sv,_=svd_rank5(target);assert nmse(sv,target)<=nmse(z,target)+1e-10
 mask=torch.arange(P,device=device)<4;r=torch.tensor([1,2,3,1],device=device);a=torch.tensor([4,5,6,7],device=device);mass=logmass.exp();u=mass*torch.expm1(b);ref=predicted(z,u,mass,mask,r,a,b);score,_=safe_predictions(z,b,logmass,mask,r,a,torch.tensor(1.,device=device));serr=float((score-ref).abs().max());assert serr<1e-10
 # Independent NumPy augmented least squares verifies ridge equations without
 # forming the normal matrix, plus decoder / candidate direct-mask verification.
 realerrors=[];decodeerrors=[];decisionerrors=[];decisionrelative=[]
 fit=torch.load(OUT/'fit.pt',weights_only=True);B=fit['B'].cuda();qs=torch.linspace(0,63,16,device=device).round().long()
 for seed in [17,29]:
  bundle=bundle_for('C',seed)
  for idx in [0,47,95,143,191,239,287]:
   d=load(files('development')[idx]);s=bundle.make(d,'C');c=contributions(d['q'],s,bundle.layers[d['key']].basis,d['log_scale']);tz,_,tb,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);F=features(bundle.reader,d['q'],c);lm=c['logzh']-c['A'].log()[:,None];z,b,co,info=fit_coeff(F,lm,tz,tb,fit['zscale'],fit['bscale']);xs=((lm-co['logmass_page_scale'][None]).exp()[...,None]*F).permute(1,0,2);ys=tz.permute(1,0,2)
   for page in [0,len(xs)//2,len(xs)-1]:
    x=xs[page].cpu().numpy();y=ys[page].cpu().numpy();scale=max(np.max(abs(y)),1e-300);reg=RIDGE*np.sum(x*x)/5;aug=np.concatenate([x,np.sqrt(reg)*np.eye(5)]);rhs=np.concatenate([y/scale,np.zeros((5,16))]);cc=np.linalg.lstsq(aug,rhs,rcond=None)[0];mine=co['coef_z_scaled'][page].cpu().numpy()/scale;er=np.max(abs(x@cc-x@mine));realerrors.append(float(er));assert er<1e-7,er
   parts=[s.keys[0].float().flatten(1),s.mean[0].float(),s.detail[0].float().flatten(1),s.stats[0].float().flatten(1)];inp=torch.cat([p/(p.detach().square().mean(-1,keepdim=True).sqrt()+1e-4) for p in parts],1);coeff=bundle.reader.page(inp).reshape(-1,5,17).bfloat16().float();pred=coeff[None,:,0]+torch.einsum('qpj,pjk->qpk',F[...,1:].float(),coeff[:,1:]);zz,uu,bb=bundle.reader(d['q'],s,c);er=float((bb-pred[...,16]).abs().max());assert er==0;decodeerrors.append(er)
   mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);score,_=safe_predictions(z[qs],b[qs],lm[qs],mask,r,a,d['dense'][qs].square().sum()/16+1e-12)
   # Canonical log-sum-exp denominators with direct signed numerator sums.
   le=lm[qs]+b[qs];ld=le.logsumexp(1);tar=z[qs].sum(1)*(-ld).exp()[:,None]
   def obj(x):
    ld=torch.where(x[None],le,lm[qs]).logsumexp(1);v=z[qs][:,x].sum(1)*(-ld).exp()[:,None];return (tar-v).square().sum(-1).mean()/(d['dense'][qs].square().sum()/16+1e-12)
   base=obj(mask)
   for j in [0,len(r)//2,int(score.argmax())]:
    x=mask.clone();x[r[j]]=False;x[a[j]]=True;v=base-obj(x);er=abs(float(v-score[j]));decisionerrors.append(er);decisionrelative.append(er/(1+abs(float(v))));assert er<1e-8+1e-7*abs(float(v))
 dump(E26/'independent_audit.json',dict(synthetic_mass_scaled_fit_max_abs=ferr,synthetic_predicted_objective_max_abs=serr,augmented_numpy_residual_fit_max_scaled_abs=max(realerrors),query_feature_decoder_max_abs=max(decodeerrors),logsumexp_decision_max_abs=max(decisionerrors),logsumexp_decision_max_relative_to_1plusabs=max(decisionrelative),decision_tolerance="1e-8 + 1e-7 * abs(reference_gain)",real_page_fits=len(realerrors),real_actions=len(decisionerrors),PCA_orthogonality_error=float((B.double().T@B.double()-torch.eye(16,device=device)).abs().max()),test_accessed=False))
 print('AUDIT OK',max(realerrors),max(decisionerrors))
if __name__=='__main__':audit()
