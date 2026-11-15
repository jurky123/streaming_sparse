from .analyze import *
@torch.no_grad()
def run():
 torch.set_num_threads(2);qs=torch.linspace(0,63,16,device='cuda').round().long();fit=torch.load(OUT/'fit.pt',weights_only=True);records=[];maxrel=0.
 for seed,indices in [(17,[47,182,231]),(29,[239,230,190])]:
  net=load_phi(seed);b=bundle_for('C',seed)
  for idx in indices:
   d,s,c,tz,tb,lm=prepare(b,files('development')[idx]);F=net(d['q'],s);z=ls_response(F,lm,tz,fit['zscale']);p=np.load(E27/f's{seed}'/f'panel_{idx:04d}.npz');r=p['remove'];a=p['add'];zz=z[qs].cpu().numpy().astype(np.longdouble);logs=lm[qs].cpu().numpy().astype(np.longdouble);be=tb[qs].cpu().numpy().astype(np.longdouble);mass=np.exp(logs);ex=np.exp(logs+be);u=ex-mass;Da=ex.sum(1);norm=float(d['dense'][qs].square().sum()/16+1e-12)
   for st,ms in enumerate(p['states']):
    Ds=np.where(ms[None],ex,mass).sum(1);H=zz[:,ms].sum(1)/Ds[:,None];E=(zz[:,~ms].sum(1)-H*u[:,~ms].sum(1)[:,None])/Da[:,None];sc=p['Adaptive64'][st]
    for j in [0,len(r)//2,int(sc.argmax())]:
     mm=ms.copy();mm[r[j]]=False;mm[a[j]]=True;den=np.where(mm[None],ex,mass).sum(1);delta=(zz[:,a[j]]-zz[:,r[j]]-H*(u[:,a[j]]-u[:,r[j]])[:,None])/den[:,None];ref=(2*(E*delta).sum(-1)-(delta**2).sum(-1)).mean()/norm;er=abs(float(ref)-float(sc[j]));rel=er/(1+abs(float(ref)));maxrel=max(maxrel,rel);records.append(dict(seed=seed,event=str(p['event']),state=st,action=j,relative_error=rel,absolute_error=er,reference=float(ref),score=float(sc[j]),strict_tolerance_pass=bool(er<1e-8+1e-7*abs(float(ref))),same_sign=bool(ref>0)==bool(sc[j]>0)))
 dump(E27/'condition_precision_audit.json',dict(checks=records,max_relative_error=maxrel,strict_exceptions=sum(not r['strict_tolerance_pass'] for r in records),sign_disagreements=sum(not r['same_sign'] for r in records),longdouble_mantissa=np.finfo(np.longdouble).nmant,test_accessed=False));print('CONDITION AUDIT',maxrel,'strict exceptions',sum(not r['strict_tolerance_pass'] for r in records),'sign disagreements',sum(not r['same_sign'] for r in records))
if __name__=='__main__':run()
