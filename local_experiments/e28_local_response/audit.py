from .run import *
@torch.no_grad()
def main():
 torch.set_num_threads(2);checks=[];B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda().double();qs=torch.linspace(0,63,16,device='cuda').round().long()
 for seed in [17,29]:
  b=bundle_for('C',seed);net=load_phi(seed);allrows=json.loads((DEST/f's{seed}'/'decisions.json').read_text());old=json.loads((E27/f's{seed}'/'decisions.json').read_text());match={(x['event'],x['policy']):x for x in old if x['method']=='Adaptive64'}
  replay=[]
  for row in allrows:
   if row['method']=='Centered-Oracle64':
    ref=match[(row['event'],row['policy'])];replay.append(dict(event=row['event'],policy=row['policy'],same_action=(row['remove'],row['add'],row['execute'])==(ref['remove'],ref['add'],ref['execute']),gain_difference=row['executed_gain']-ref['executed_gain']))
  dump(DEST/f'replay_s{seed}.json',replay)
  for idx in [0,47,144,230]:
   path=files('development')[idx];d,s,c,t,beta,lm=prepare(b,path);F=net(d['q'],s);ap=summary_output(c,s,b.layers[d['key']].basis);local=(d['ev']['page_output'].cuda()-ap)@B;center=(ap-c['o'][:,None])@B
   coef,pred,_=fitmany(F,lm,local)
   for p in [0,d['hp']-1,d['hp']]:
    X=F[:,p].cpu().numpy();w=np.exp((lm[:,p]-lm[:,p].max()).cpu().numpy());X=X*w[:,None];Y=local[:,p].cpu().numpy()*w[:,None];lam=RIDGE*np.trace(X.T@X)/5
    CC=np.linalg.lstsq(np.concatenate([X,np.sqrt(lam)*np.eye(5)]),np.concatenate([Y,np.zeros((5,16))]),rcond=None)[0]
    error=np.max(abs(CC-coef[p].cpu().numpy()))/max(1.,np.max(abs(CC)))
    assert error<1e-7
    checks.append(dict(seed=seed,event=str(path.relative_to(OUT)),page=p,augmented_LS_relative_error=float(error)))
   mask,r,a=route(d['q'][qs],s,b.layers[d['key']],d['own']);z=reconstruct(pred,beta,lm,center);score,_=safe_predictions(z[qs],beta[qs],lm[qs],mask,r,a,d['dense'][qs].square().sum()/16+1e-12)
   # Independent long-double direct output objectives for chosen action.
   zz=z[qs].cpu().numpy().astype(np.longdouble);mm=np.exp(lm[qs].cpu().numpy().astype(np.longdouble));ee=np.exp((lm[qs]+beta[qs]).cpu().numpy().astype(np.longdouble));uu=ee-mm;m=mask.cpu().numpy();j=int(score.argmax());new=m.copy();new[int(r[j])]=False;new[int(a[j])]=True
   target=zz.sum(1)/ee.sum(1)[:,None]
   h=zz[:,m].sum(1)/np.where(m,ee,mm).sum(1)[:,None];h2=zz[:,new].sum(1)/np.where(new,ee,mm).sum(1)[:,None];diff=h2-h;E=target-h;norm=np.longdouble(float(d['dense'][qs].square().sum()/16+1e-12));true=((2*E*diff-diff*diff).sum(-1)).mean()/norm
   rr=int(r[j]);aa=int(a[j]);N=d['N'].cpu().numpy().astype(np.longdouble);Z=d['Z'].cpu().numpy().astype(np.longdouble);NH=c['nh'].cpu().numpy().astype(np.longdouble);ZH=c['zh'].cpu().numpy().astype(np.longdouble);OD=d['dense'].cpu().numpy().astype(np.longdouble)
   def exact_error(mm):
    oo=np.where(mm[None,:,None],N,NH).sum(1)/np.where(mm[None],Z,ZH).sum(1)[:,None]
    return ((OD-oo)**2).sum()/((OD**2).sum()+np.longdouble(1e-12))
   gt=exact_error(m)-exact_error(new);saved=next(x for x in allrows if x['event']==str(path.relative_to(OUT)) and x['method']=='Local-Oracle64' and x['policy']=='Set')
   assert saved['remove']==rr and saved['add']==aa
   checks.append(dict(seed=seed,event=str(path.relative_to(OUT)),true_gain_longdouble=float(gt),saved_gain=saved['proposal_gain'],true_gain_difference=abs(float(gt)-saved['proposal_gain']),own_preserved=bool(new[d['own']]),count_preserved=int(new.sum())==int(m.sum())))
   err=abs(float(true)-float(score[j]));checks.append(dict(seed=seed,event=str(path.relative_to(OUT)),score=float(score[j]),longdouble=float(true),score_error=err,relative_error=err/max(1.,abs(float(true))),count=int(mask.sum()),own=bool(mask[d['own']])))
 dump(DEST/'independent_audit.json',dict(checks=checks,test_accessed=False))
 print('AUDIT DONE',flush=True)
if __name__=='__main__':main()
