from .run import *

@torch.no_grad()
def audit():
 torch.set_num_threads(2);torch.manual_seed(1701)
 # Synthetic exact online-centred identity and projected teacher equivalence.
 Q,P,V=64,12,128;N=torch.randn(Q,P,V,dtype=torch.double);Z=torch.rand(Q,P,dtype=torch.double)+.1;nh=torch.randn_like(N);zh=torch.rand_like(Z)+.1;A=zh.sum(1);o0=nh.sum(1)/A[:,None];t=(N-nh-o0[:,None]*(Z-zh)[...,None])/A[:,None,None];u=(Z-zh)/A[:,None];B=torch.linalg.qr(torch.randn(V,16,dtype=torch.double)).Q;od=N.sum(1)/Z.sum(1)[:,None];err=0.;projerr=0.
 for _ in range(16):
  mask=torch.zeros(P,dtype=torch.bool);mask[0]=True;mask[torch.randperm(P-1)[:3]+1]=True;direct=torch.where(mask[None,:,None],N,nh).sum(1)/torch.where(mask[None],Z,zh).sum(1)[:,None];ident=o0+t[:,mask].sum(1)/(1+u[:,mask].sum(1))[:,None];err=max(err,float((ident-direct).abs().max()));p=(t@B).sum(1)/(1+u.sum(1))[:,None]-(t@B)[:,mask].sum(1)/(1+u[:,mask].sum(1))[:,None];projerr=max(projerr,float((p-(od-direct)@B).abs().max()))
 assert err<1e-12 and projerr<1e-12
 B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda().double();qs=torch.linspace(0,63,16,device='cuda').round().long();paths=files('development');errs={k:0. for k in ['P1','P2','P3']};count=0;rows=[]
 for group in ['A','C']:
  for seed in [17,29]:
   b=bundle_for(group,seed)
   for idx in [0,23,47,71,95,119,143,167,191,215,239,287]:
    d=load(paths[idx]);s=b.make(d,group);m=b.layers[d['key']];c=contributions(d['q'],s,m.basis,d['log_scale']);mask,r,a=route(d['q'][qs],s,m,d['own']);ev=d['ev'];logz=ev['page_logz'].cuda().double();op=ev['page_output'].cuda().double();sl=c['logit'].logsumexp(-1)+math.log(16);nu=s.mean[0].double()[:,None]+torch.einsum('ag,pak,dk->pgd',haar(d['q']).double(),s.detail[0].double(),m.basis[0].bfloat16().double());so=torch.einsum('qpg,pgd->qpd',torch.softmax(c['logit'],dim=-1),nu);dense=torch.einsum('qp,qpd->qd',torch.softmax(logz,dim=1),op)
    def canonical(x):
     weights=torch.softmax(torch.where(x[None],logz,sl),dim=1);return torch.einsum('qp,qpd->qd',weights,torch.where(x[None,:,None],op,so))
    base=dense-canonical(mask)
    for name,q,Buse in [('P1',qs,B),('P2',qs,None),('P3',torch.arange(64,device='cuda'),None)]:
     scores=project_gains(c,d,mask,r,a,q,Buse);js=sorted(set([0,31,63,127,255,511,int(scores.argmax())]));js=[j for j in js if j<len(r)];energy=dense[q].square().sum()+1e-12;b0=base[q] if Buse is None else base[q]@Buse
     for j in js:
      x=mask.clone();x[r[j]]=False;x[a[j]]=True;res=(dense-canonical(x))[q];res=res if Buse is None else res@Buse;g=(b0.square().sum()-res.square().sum())/energy;er=abs(float(g-scores[j]));errs[name]=max(errs[name],er);count+=1
    rows.append(dict(group=group,seed=seed,event=str(paths[idx].relative_to(OUT))))
 assert max(errs.values())<1e-8,errs
 dump(E25/'independent_audit.json',dict(synthetic_online_identity_max_abs=err,synthetic_projected_identity_max_abs=projerr,canonical_logsoftmax_gain_max_abs=errs,teacher_actions_checked=count,events=rows,test_accessed=False,teacher_path='Canonical page logZ and normalized output + stable R4 softmax; independent of stored scaled N/Z'))
 print('AUDIT',errs,count,flush=True)
if __name__=='__main__':audit()
