from .run import *
@torch.no_grad()
def check():
 torch.set_num_threads(2);qs=torch.linspace(0,63,16,device='cuda').round().long();maxerr=0.;checked=0;rows=[]
 for group in ['A','C']:
  for seed in [17,29]:
   b=bundle_for(group,seed)
   for idx in [0,71,143,215,287]:
    p=np.load(E25/f'{group}_s{seed}'/f'panel_{idx:04d}.npz');event=str(p['event']);d=load(OUT/event);s=b.make(d,group);c=contributions(d['q'],s,b.layers[d['key']].basis,d['log_scale']);cp={n:x[qs] for n,x in c.items()};z,u,beta=b.reader(d['q'][qs],s,cp);mass=cp['mass'];ex=mass*beta.double().exp();target=z.sum(1)/ex.sum(1)[:,None];norm=d['dense'][qs].square().sum()/16+1e-12;rr=p['remove'];aa=p['add']
    def predobj(mask):
     den=torch.where(mask[None],ex,mass).sum(1);assert (den>0).all();return (target-z[:,mask].sum(1)/den[:,None]).square().sum(-1).mean()/norm
    for st_i,state in enumerate(p['states']):
     mask=torch.as_tensor(state,device='cuda');assert mask[d['own']];assert int(mask.sum())==max(1,int(len(mask)*.1));assert mask[rr].all() and (~mask[aa]).all();base=predobj(mask)
     for j in [0,len(rr)//2,len(rr)-1]:
      other=mask.clone();other[rr[j]]=False;other[aa[j]]=True;direct=base-predobj(other);stored=p['predicted_gain'][st_i,j];er=abs(float(direct)-stored);maxerr=max(maxerr,er);assert er<=1e-8+1e-6*abs(stored),(er,stored);checked+=1
    rows.append(dict(group=group,seed=seed,event=event))
 dump(E25/'panel_audit.json',dict(max_abs_error=maxerr,actions_checked=checked,events=rows,all_common_pairs_legal=True,test_accessed=False));print('PANEL AUDIT',maxerr,checked)
if __name__=='__main__':check()
