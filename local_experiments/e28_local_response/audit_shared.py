from .run import *
@torch.no_grad()
def main():
 torch.set_num_threads(2);checks=[];B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda().double()
 for seed in [17,29]:
  bundle=bundle_for('C',seed);net=load_phi(seed);ids=json.loads((DEST/f's{seed}'/'identities.json').read_text())
  for idx in [0,230]:
   event=str(files('development')[idx].relative_to(OUT));ident=next(x for x in ids if any(o['event']==event and o['page']==0 for o in x['occurrences']));Fs=[];Ls=[];Ys=[];Ts=[];held=None
   for j,o in enumerate(ident['occurrences']):
    p=OUT/o['event'];d,s,c,t,bt,lm=prepare(bundle,p);pp=o['page'];kv=torch.load(p.parent/d['ev']['kv_file'],weights_only=True);assert digest(kv['ids'][pp],kv['k'][pp],kv['v'][pp])==ident['raw_sha'];assert digest(*(getattr(s,n)[0,pp] for n in ['keys','mean','detail','stats']))==ident['summary_sha']
    F=net(d['q'],s);ap=summary_output(c,s,bundle.layers[d['key']].basis);Y=(d['ev']['page_output'].cuda()-ap)@B;Fs.append(F[:,pp:pp+1]);Ls.append(lm[:,pp:pp+1]);Ys.append(Y[:,pp:pp+1]);Ts.append(t[:,pp:pp+1])
    if o['event']==event:held=j
   for mode in ['Shared','LOTO']:
    use=list(range(len(Fs))) if mode=='Shared' else [i for i in range(len(Fs)) if i!=held];assert mode!='LOTO' or held not in use
    F=torch.cat([Fs[i] for i in use]);lm=torch.cat([Ls[i] for i in use]);mx=lm.max(0).values;X=((lm-mx).exp()[...,None]*F)[:,0].cpu().numpy()
    for target in ['Centered','Local']:
     yy=torch.cat([(Ts if target=='Centered' else Ys)[i] for i in use]);co,_,_=(fitcenter if target=='Centered' else fitmany)(F,lm,yy);Y=yy[:,0].cpu().numpy()
     if target=='Local':Y=Y*np.exp((lm[:,0]-mx[0]).cpu().numpy())[:,None]
     scale=max(np.max(abs(Y)),1e-300);Y=Y/scale;lam=RIDGE*np.trace(X.T@X)/5;ref=np.linalg.lstsq(np.concatenate([X,np.sqrt(lam)*np.eye(5)]),np.concatenate([Y,np.zeros((5,16))]),rcond=None)[0];pred=co[0].cpu().numpy()/scale;err=float(np.max(abs(ref-pred))/max(np.max(abs(ref)),1e-300));checks.append(dict(seed=seed,event=event,page=0,mode=mode,target=target,occurrences=len(Fs),fitted_queries=64*len(use),relative_coefficient_error=err,withheld_event_in_fit=held in use));assert err<1e-6
 dump(DEST/'shared_audit.json',checks);print('SHARED AUDIT DONE',flush=True)
if __name__=='__main__':main()
