"""E30-A: frozen-response beta capacity, context reuse and decision sensitivity."""
from ..e29_local_reader.core import *
from ..e28_local_response.run import digest
from ..e26_oracle_coefficients.report import stablecorr
import collections
DEST=ROOT/'artifacts/e30_beta_diagnostics_v1'
METHODS=['Predicted','OracleCoeff64','Shared','LOTO','True','Candidate-True','Background-True']

def fit_beta(F,lm,beta,scale):
 w=(lm-lm.max(0).values[None]).exp()
 C,_,audit=ridge_fit((w[...,None]*F).permute(1,0,2),(w*beta).T[...,None],scale)
 return C,torch.einsum('qpf,pfd->qpd',F,C)[...,0],audit

def local_scores(rp,beta,lm,center,mask,r,a,norm):
 # Algebraically identical local-response model, but normalize each mixture
 # separately. Never form exp(beta), z, or an overflowing unnormalized sum.
 rp=rp.double();beta=beta.double();v=center+rp;le=lm+beta
 target=(le.softmax(1)[...,None]*v).sum(1)
 current_log=torch.where(mask[None],le,lm)
 H=(current_log.softmax(1)[...,None]*(center+mask[None,:,None]*rp)).sum(1)
 E=target-H;scores=[]
 for st in range(0,len(r),32):
  rr,aa=r[st:st+32],a[st:st+32];ms=mask[None].expand(len(rr),-1).clone();idx=torch.arange(len(rr),device=rp.device);ms[idx,rr]=False;ms[idx,aa]=True
  logs=torch.where(ms[None],le[:,None],lm[:,None]);w=logs.softmax(-1)
  Hnew=torch.bmm(w,center)+torch.bmm(w*ms[None],rp)
  delta=Hnew-H[:,None];scores.append((2*(E[:,None]*delta).sum(-1)-delta.square().sum(-1)).mean(0)/norm)
 score=torch.cat(scores)
 logDa=le.logsumexp(1);lp=lm-logDa[:,None];extra=lp.max().clamp_min(0).detach()
 net=(le-logDa[:,None]-extra).exp()[...,None]*(v-target[:,None])-(lp-extra).exp()[...,None]*(center-target[:,None])
 energy=net.square().sum(-1).mean(0);ind=energy[a]-energy[r]
 # The omitted exp(2*extra) is one positive constant for all actions/Queries.
 assert torch.isfinite(score).all() and torch.isfinite(ind).all()
 return score,ind

def interaction(z,beta,lm,mask,r,a,norm):
 shift=torch.maximum(lm.max(1).values,(lm+beta).max(1).values)
 shift=torch.maximum(shift,z.abs().amax((1,2)).clamp_min(1e-300).log())
 m=(lm-shift[:,None]).exp();e=(lm+beta-shift[:,None]).exp();zz=z*(-shift).exp()[:,None,None]
 u=e-m;Da=e.sum(1);Ds=torch.where(mask[None],e,m).sum(1);od=zz.sum(1)/Da[:,None]
 net=zz-od[:,None]*u[...,None];E=net[:,~mask].sum(1)/Ds[:,None];outI=[];outB=[]
 for i in range(0,len(r),32):
  de=(net[:,r[i:i+32]]-net[:,a[i:i+32]])/Ds[:,None,None]
  outI.append((-2*(E[:,None]*de).sum(-1)).mean(0)/norm);outB.append(de.square().sum(-1).mean(0)/norm)
 return torch.cat(outI),torch.cat(outB)

def weighted_metrics(bb,bt,lm,idx):
 b=bb[:,idx].double();t=bt[:,idx];l=lm[:,idx];err=b-t
 out=dict(beta_mae=float(err.abs().mean()),beta_rmse=float(err.square().mean().sqrt()))
 for name,logw in [('summary',l),('true',l+t),('predicted',l+b)]:
  w=torch.softmax(logw.flatten(),0).reshape_as(logw)
  out[name+'_weighted_beta_mae']=float((w*err.abs()).sum());out[name+'_weighted_beta_rmse']=float((w*err.square()).sum().sqrt())
 # exp(logmass+beta) is exact-mass relative to total summary mass; avoid exp(beta).
 sh=torch.maximum((l+b).max(),(l+t).max());pred=(l+b-sh).exp();true=(l+t-sh).exp()
 log1=(pred-true).abs().sum().log()+sh-torch.logsumexp((l+t).flatten(),0)
 log2=.5*((pred-true).square().sum().log()+2*sh-torch.logsumexp((2*(l+t)).flatten(),0))
 out['quality_mass_l1_relative']=float(log1.exp()) if log1<709 else None
 out['quality_mass_l2_relative']=float(log2.exp()) if log2<709 else None
 out['quality_mass_log_l1_relative']=float(log1) if torch.isfinite(log1) else None
 out['quality_mass_log_l2_relative']=float(log2) if torch.isfinite(log2) else None
 out['quality_mass_exact_zero_error']=bool((pred==true).all())
 return out

def sensitivity(rec,qs,base_score):
 rp=rec['rp'][qs].double();bp=rec['bp'][qs].double();bt=rec['bt'][qs];lm=rec['lm'][qs];center=rec['center'][qs]
 j=int(base_score.argmax());rr=rec['r'][j:j+1];aa=rec['a'][j:j+1]
 with torch.enable_grad():
  var=bp.detach().clone().requires_grad_(True);sc,_=local_scores(rp,var,lm,center,rec['mask'],rr,aa,rec['norm'])
  grad=torch.autograd.grad(sc.sum(),var)[0].detach()
 influence=(grad*(bt-bp)).sum(0);strength=(grad*(bt-bp)).abs().sum(0);order=torch.argsort(strength,descending=True);den=strength.sum().clamp_min(1e-300)
 result=dict(proposal=int(j),true_gain=float(rec['g'][j]),executed=bool(base_score[j]>0),predicted_score=float(base_score[j]),linear_total_delta=float(influence.sum()),gradient_abs_max=float(grad.abs().max()))
 for k in [1,4,8]:result[f'top{k}_linear_absolute_share']=float(strength[order[:k]].sum()/den)
 rows=[];bfix=bp.clone()
 for rank,p in enumerate(order[:8].tolist(),1):
  single=bp.clone();single[:,p]=bt[:,p];sg,_=local_scores(rp,single,lm,center,rec['mask'],rr,aa,rec['norm'])
  bfix[:,p]=bt[:,p];cg,_=local_scores(rp,bfix,lm,center,rec['mask'],rr,aa,rec['norm'])
  rows.append(dict(rank=rank,page=p,candidate=bool(rec['candidate'][p]),history=p<rec['hp'],beta_mae=float((bp[:,p]-bt[:,p]).abs().mean()),linear_delta=float(influence[p]),linear_absolute=float(strength[p]),single_corrected_score=float(sg[0]),cumulative_corrected_score=float(cg[0])))
 allsc,_=local_scores(rp,bt,lm,center,rec['mask'],rr,aa,rec['norm']);result['full_true_beta_same_action_score']=float(allsc[0]);result['top8_same_action_score']=rows[-1]['cumulative_corrected_score'];result['sensitivity_fp64_base_score']=float(sc[0].detach())
 # Verify analytic derivative dt/dbeta and du/dbeta with central finite difference.
 p=int(order[0]);q=int(grad[:,p].abs().argmax());eps=1e-5;plus=bp.clone();minus=bp.clone();plus[q,p]+=eps;minus[q,p]-=eps
 ps,_=local_scores(rp,plus,lm,center,rec['mask'],rr,aa,rec['norm']);ms,_=local_scores(rp,minus,lm,center,rec['mask'],rr,aa,rec['norm'])
 fd=(ps[0]-ms[0])/(2*eps);ag=grad[q,p];result['gradient_fd_abs_error']=float((fd-ag).abs());result['gradient_fd_relative_error']=float((fd-ag).abs()/(fd.abs()+ag.abs()).clamp_min(1e-12))
 return result,rows,grad,strength

@torch.no_grad()
def run(seed,limit_groups=None):
 torch.set_num_threads(2);dest=DEST/f's{seed}';dest.mkdir(parents=True,exist_ok=True)
 assert not (dest/'complete.json').exists(), 'completed output is immutable'
 bundle=bundle_for('C',seed);phi=load_phi(seed);reader=LocalReader().cuda().eval();reader.load_state_dict(torch.load(E29/f's{seed}'/'selected.pt',map_location='cuda',weights_only=False)['model'])
 for x in [phi,reader]:x.eval()
 qs=torch.linspace(0,63,16,device='cuda').round().long();scale=torch.load(OUT/'fit.pt',weights_only=True)['bscale'];start=time.time()
 identpath=ROOT/'artifacts/e28_local_response_v1'/f's{seed}'/'identities.json';ident=json.loads(identpath.read_text());ig=collections.defaultdict(list)
 for item in ident:ig[tuple(item['group'])].append(item)
 groups=collections.defaultdict(list)
 for idx,path in enumerate(files('development')):
  ev=torch.load(path,map_location='cpu',weights_only=True);groups[(ev['trajectory'],ev['layer'],ev['head'])].append((idx,path))
 decisions=[];info=[];conditions=[];fits=[];audit=[];sens=[];senspages=[];count=0
 for gi,(group,paths) in enumerate(groups.items()):
  if limit_groups is not None and gi>=limit_groups:break
  records=[];emap={}
  for idx,path in paths:
   d,s,c,rt,bt,lm,center=prepare_local(bundle,path);F=phi(d['q'],s);rp,bp=reader(s,F);C,oracle,fa=fit_beta(F,lm,bt,scale)
   mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);panel=np.load(E25/f'C_s{seed}'/f'panel_{idx:04d}.npz')
   assert np.array_equal(mask.cpu(),panel['states'][0]);assert np.array_equal(r.cpu(),panel['initial_remove']);assert np.array_equal(a.cpu(),panel['initial_add'])
   candidate=torch.zeros_like(mask);candidate[r]=True;candidate[a]=True
   ap=summary_output(c,s,bundle.layers[d['key']].basis);tz=reconstruct(d['ev']['page_output'].cuda().double()-ap,bt,lm,ap-c['o'][:,None]);norm=d['dense'][qs].square().sum()/16+1e-12
   ti,tb=interaction(tz,bt,lm,mask,r,a,d['dense'].square().sum()/64+1e-12);ti16,tb16=interaction(tz[qs],bt[qs],lm[qs],mask,r,a,norm)
   ev=d['ev'];event=str(path.relative_to(OUT));rec=dict(idx=idx,event=event,ev=ev,path=path,F=F,rp=rp,bp=bp,bt=bt,lm=lm,center=center,mask=mask,r=r,a=a,hp=d['hp'],candidate=candidate,norm=norm,J=float(objective(c,d['N'],d['Z'],d['dense'],mask)),g=torch.as_tensor(panel['true_initial_gain'],device='cuda'),panel={k:panel[k] for k in ['states','remove','add','true_gain']},betas={'Predicted':bp,'OracleCoeff64':oracle,'Shared':oracle.clone(),'LOTO':oracle.clone(),'True':bt,'Candidate-True':torch.where(candidate[None],bt,bp),'Background-True':torch.where(candidate[None],bp,bt)},trueI=ti,trueB=tb,trueI16=ti16,trueB16=tb16,summary_sha=[digest(*(getattr(s,n)[0,p] for n in ['keys','mean','detail','stats'])) for p in range(d['hp'])])
   # Recompute raw identity on one registered page/event; all summary hashes checked below.
   kv=torch.load(path.parent/ev['kv_file'],map_location='cpu',weights_only=True);rec['raw_sample_sha']=digest(kv['ids'][0],kv['k'][0],kv['v'][0]) if d['hp'] else None
   emap[event]=len(records);records.append(rec);fits.append(dict(seed=seed,event=event,mode='OracleCoeff64',**fa))
  signatures=collections.defaultdict(list);matched=0
  for item in ig[group]:
   occ=[(emap[o['event']],o['page']) for o in item['occurrences']]
   for i,p in occ:
    assert records[i]['summary_sha'][p]==item['summary_sha'];matched+=1
    if p==0:assert records[i]['raw_sample_sha']==item['raw_sha']
   assert len(occ)>1,'LOTO undefined for singleton history page'
   signatures[tuple(i for i,p in occ)].append(occ)
  assert matched==sum(r['hp'] for r in records)
  for sig,items in signatures.items():
   ix=[torch.tensor([it[j][1] for it in items],device='cuda') for j in range(len(sig))]
   fs=[records[i]['F'][:,pp] for i,pp in zip(sig,ix)];ls=[records[i]['lm'][:,pp] for i,pp in zip(sig,ix)];ys=[records[i]['bt'][:,pp] for i,pp in zip(sig,ix)]
   co,_,fa=fit_beta(torch.cat(fs),torch.cat(ls),torch.cat(ys),scale);fits.append(dict(seed=seed,group=str(group),mode='Shared',pages=len(items),occurrences=len(sig),**fa))
   for j,(i,pp) in enumerate(zip(sig,ix)):
    records[i]['betas']['Shared'][:,pp]=torch.einsum('qpf,pfd->qpd',fs[j],co)[...,0]
    others=[k for k in range(len(sig)) if k!=j];cl,_,fa=fit_beta(torch.cat([fs[k] for k in others]),torch.cat([ls[k] for k in others]),torch.cat([ys[k] for k in others]),scale)
    records[i]['betas']['LOTO'][:,pp]=torch.einsum('qpf,pfd->qpd',fs[j],cl)[...,0];fits.append(dict(seed=seed,event=records[i]['event'],mode='LOTO',pages=len(items),fit_occurrences=len(others),**fa))
  for rec in records:
   idx=rec['idx'];event=rec['event'];lm=rec['lm'];bt=rec['bt'];r=rec['r'];a=rec['a'];mask=rec['mask'];g=rec['g'];panel=rec['panel'];scoresout={};initialout={};interactions={};metadata=dict(seed=seed,event=event,**{k:rec['ev'][k] for k in ['trajectory','layer','head','chunk','step','tile']})
   for method,bb in rec['betas'].items():
    z=reconstruct(rec['rp'][qs],bb[qs],lm[qs],rec['center'][qs]);sc,ind=local_scores(rec['rp'][qs],bb[qs],lm[qs],rec['center'][qs],mask,r,a,rec['norm']);initialout[method]=sc.cpu().numpy()
    ii,ee=interaction(z,bb[qs],lm[qs],mask,r,a,rec['norm']);interactions[method]=ii.cpu().numpy()
    for policy,score in [('Set',sc),('Independent',ind)]:
     j=int(score.argmax());ex=bool(score[j]>0);new=mask.clone()
     if ex:new[r[j]]=False;new[a[j]]=True
     assert int(new.sum())==int(mask.sum()) and bool(new[rec['ev']['own']])
     decisions.append(dict(**metadata,method=method,policy=policy,initial=rec['J'],oracle_gain=max(0.,float(g.max())),proposal_gain=float(g[j]),hindsight_gain=max(0.,float(g[j])),executed_gain=float(g[j]) if ex else 0.,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(score[j])))
    for qname,qq in [('M16',qs),('Q64',torch.arange(64,device='cuda'))]:
     for region,select in [('all',torch.ones_like(mask)),('candidate',rec['candidate']),('background',~rec['candidate'])]:
      info.append(dict(**metadata,method=method,queries=qname,region=region,**weighted_metrics(bb[qq],bt[qq],lm[qq],select)))
    actionrow=dict(**metadata,method=method,queries='M16',region='action_panel',gain_pearson=stablecorr(sc.cpu().numpy(),g.cpu().numpy()),gain_spearman=float(spearmanr(sc.cpu().numpy(),g.cpu().numpy()).statistic),interaction_numerically_valid=bool(torch.isfinite(ii).all() and torch.isfinite(ee).all()),interaction_sign_accuracy=float(((ii>0)==(rec['trueI']>0)).double().mean()) if torch.isfinite(ii).all() else None,interaction_sign_accuracy_M16=float(((ii>0)==(rec['trueI16']>0)).double().mean()) if torch.isfinite(ii).all() else None,energy_nmse=nmse(ee,rec['trueB']) if torch.isfinite(ee).all() else None,gain_positive_accuracy=float(((sc>0)==(g>1e-12)).double().mean()),gain_positive_recall=float(((sc>0)&(g>1e-12)).sum()/((g>1e-12).sum().clamp_min(1))),best_action_regret=float(g.max().clamp_min(0)-g[int(sc.argmax())]));info.append(actionrow)
    rr=torch.as_tensor(panel['remove'],device='cuda');aa=torch.as_tensor(panel['add'],device='cuda');ps=np.stack([local_scores(rec['rp'][qs],bb[qs],lm[qs],rec['center'][qs],torch.as_tensor(st,device='cuda'),rr,aa,rec['norm'])[0].cpu().numpy() for st in panel['states']]);scoresout[method]=ps;conditions.append(dict(**metadata,method=method,**conditional_metrics(panel['true_gain'],ps)))
   sr,sp,grad,strength=sensitivity(rec,qs,torch.as_tensor(initialout['Predicted'],device='cuda'));sens.append(dict(**metadata,**sr));senspages.extend(dict(**metadata,**p) for p in sp)
   # Exact replay of saved E29 baseline/TrueBeta action and counterfactual scores.
   prior=np.load(E29/f's{seed}'/f'panel_{idx:04d}.npz');err=max(float(np.max(np.abs(scoresout[x]-prior[y])/np.maximum(1.,np.abs(prior[y])))) for x,y in [('Predicted','Local-Online'),('True','Local-TrueBeta')]);# Numerical reformulation may differ near cancellation; audit actions and longdouble independently.
   audit.append(dict(**metadata,replay_relative_error=err,history_summary_hashes_checked=rec['hp'],raw_page0_checked=True,finite_all=True,max_abs_fitted_beta=float(rec['betas']['LOTO'].abs().max())))
   np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=event,true_gain=panel['true_gain'],true_initial_gain=g.cpu().numpy(),states=panel['states'],remove=panel['remove'],add=panel['add'],initial_remove=r.cpu().numpy(),initial_add=a.cpu().numpy(),trueI=rec['trueI'].cpu().numpy(),trueB=rec['trueB'].cpu().numpy(),gradient_beta=grad.cpu().numpy(),sensitivity_strength=strength.cpu().numpy(),**scoresout,**{'initial_'+k:v for k,v in initialout.items()},**{'interaction_'+k:v for k,v in interactions.items()})
   torch.save(dict(event=event,F=rec['F'].cpu(),logmass=lm.cpu(),predicted_response=rec['rp'].cpu(),center=rec['center'].cpu(),beta_true=bt.cpu(),betas={k:v.cpu() for k,v in rec['betas'].items()}),dest/f'beta_{idx:04d}.pt')
   count+=1
  for name,data in [('decisions',decisions),('information',info),('conditional',conditions),('fits',fits),('audit',audit),('sensitivity',sens),('sensitivity_pages',senspages)]:dump(dest/(name+'.json'),data)
  dump(dest/'status.json',dict(seed=seed,groups_done=gi+1,events=count,seconds=time.time()-start));print('GROUP',seed,gi+1,'events',count,'seconds',round(time.time()-start),flush=True)
 if limit_groups is None:
  assert count==288
  dump(dest/'complete.json',dict(seed=seed,events=count,seconds=time.time()-start,test_accessed=False,training=False,reader_sha=sha(E29/f's{seed}'/'selected.pt'),C_sha=sha(checkpoint('C',seed)),phi_sha=sha(E27/f's{seed}'/'selected.pt'),identity_sha=sha(identpath),protocol_sha=sha(DEST/'protocol.json')))
 print('DONE',seed,count,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--limit-groups',type=int);args=p.parse_args();run(args.seed,args.limit_groups)
