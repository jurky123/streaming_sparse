"""Conditional E27-B: frozen summaries and learned phi, original page capacity."""
from .analyze import *

class Coefficients(nn.Module):
 def __init__(self):
  super().__init__();self.page=nn.Sequential(nn.Linear(712,64),nn.GELU(),nn.Linear(64,85));nn.init.normal_(self.page[-1].weight,std=.001);nn.init.zeros_(self.page[-1].bias)
 def forward(self,s,F,mass):
  co=self.page(page_features(s)).reshape(-1,5,17).bfloat16().float();val=co[None,:,0]+torch.einsum('qpj,pjk->qpk',F[...,1:].float(),co[:,1:]);return mass[...,None]*val[...,:16],val[...,16]

def rms_floor(x,dim,floor):
 scale=x.abs().amax(dim,keepdim=True).clamp_min(floor);return (scale*(x/scale).square().mean(dim,keepdim=True).sqrt()).clamp_min(floor)

@torch.no_grad()
def eval_b(model,phi,bundle,seed,paths):
 qs=torch.linspace(0,63,16,device='cuda').round().long();rows=[]
 for path in paths:
  d,s,c,zt,bt,lm=prepare(bundle,path);F=phi(d['q'],s);mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);J=float(objective(c,d['N'],d['Z'],d['dense'],mask));z,beta=model(s,F[qs],c['mass'][qs]);norm=d['dense'][qs].square().sum()/16+1e-12
  for name,be in [('PredictedBeta',beta),('TrueBeta',bt[qs])]:
   score,ind=safe_predictions(z,be,lm[qs],mask,r,a,norm)
   for policy,sc in [('Set',score),('Independent',ind)]:
    j=int(sc.argmax());ex=bool(sc[j]>0);true=float(g[j]);rows.append(dict(seed=seed,event=str(path.relative_to(OUT)),trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],method=name,policy=policy,initial=J,oracle_gain=max(0.,float(g.max())),proposal_gain=true,hindsight_gain=max(0.,true),executed_gain=true if ex else 0.,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(sc[j])))
 return rows

def train_b(seed):
 assert json.loads((E27/'B_gate.json').read_text())['passed'];torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);dest=E27/f'B_s{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'training_complete.json').exists():return
 bundle=bundle_for('C',seed);phi=load_phi(seed)
 for p in itertools.chain(bundle.parameters(),phi.parameters()):p.requires_grad_(False)
 frozen={n:t.cpu().clone() for n,t in bundle.layers.state_dict().items()};frozen_phi={n:t.cpu().clone() for n,t in phi.state_dict().items()}
 model=Coefficients().cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);fit=torch.load(OUT/'fit.pt',weights_only=True);qs=torch.linspace(0,63,16,device='cuda').round().long();rng=np.random.default_rng(seed);trainfiles=files('train');devfiles=files('development');curves=[];cks=[];start=time.time();torch.save(model.state_dict(),dest/'initial.pt')
 for epoch in range(4):
  for j,idx in enumerate(rng.permutation(len(trainfiles))):
   d,s,c,zt,bt,lm=prepare(bundle,trainfiles[idx])
   with torch.no_grad():
    F=phi(d['q'],s);mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);ff,_=RouteOps.features(d['q'][qs][None],s);risk=(ff*bundle.layers[d['key']].theta[:,None,None]).sum(-1).mean(1)[0].cpu().numpy();mask,r,a=getstate(c,d,mask,r,a,(epoch*len(trainfiles)+j)%10,risk,rng);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);zs=rms_floor(zt[qs],(0,2),fit['zscale']);bs=rms_floor(bt[qs],(0,),fit['bscale']);gs=rms_floor(g,(0,),fit['gscale']).reshape(())
   z,beta=model(s,F[qs],c['mass'][qs]);res=nn.functional.smooth_l1_loss(z/zs,zt[qs]/zs)+nn.functional.smooth_l1_loss(beta/bs,bt[qs]/bs);pg,_=safe_predictions(z,beta,lm[qs],mask,r,a,d['dense'].square().sum()/64+1e-12);target=torch.softmax(torch.cat([g,g.new_zeros(1)])/(.3*gs),0);rank=-(target*torch.log_softmax(torch.cat([pg,pg.new_zeros(1)])/gs,0)).sum();setloss=nn.functional.smooth_l1_loss(pg/gs,g/gs)+.1*rank;loss=res+setloss;assert torch.isfinite(loss)
   opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();curves.append(dict(epoch=epoch,update=epoch*len(trainfiles)+j,residual=float(res),set_loss=float(setloss),loss=float(loss),gradnorm=float(grad),gain_scale=float(gs)))
   if j%100==0:dump(dest/'status.json',dict(epoch=epoch,update=j,seconds=time.time()-start));print('B',seed,epoch,j,float(loss),flush=True)
  rows=eval_b(model,phi,bundle,seed,devfiles);rr=[r for r in rows if r['method']=='PredictedBeta' and r['policy']=='Set'];gain=sum(r['executed_gain'] for r in rr);ora=sum(r['oracle_gain'] for r in rr);meta=dict(epoch=epoch,gain=gain,recovery=gain/ora);cks.append(meta);torch.save(dict(model=model.state_dict(),**meta),dest/f'epoch{epoch}.pt');dump(dest/f'development_epoch{epoch}.json',rows);dump(dest/'curves.json',curves);print('B EPOCH',seed,meta,flush=True)
 best=max(cks,key=lambda x:x['gain']);torch.save(torch.load(dest/f'epoch{best["epoch"]}.pt',map_location='cpu',weights_only=False),dest/'selected.pt');assert all(torch.equal(v.cpu(),frozen[n]) for n,v in bundle.layers.state_dict().items());assert all(torch.equal(v.cpu(),frozen_phi[n]) for n,v in phi.state_dict().items());dump(dest/'training_complete.json',dict(seed=seed,updates=len(curves),best=best,checkpoints=cks,seconds=time.time()-start,frozen_C_exact=True,frozen_phi_exact=True,C_sha=sha(checkpoint('C',seed)),phi_sha=sha(E27/f's{seed}'/'selected.pt'),parameters=sum(p.numel() for p in model.parameters()),test_accessed=False));print('B DONE',seed,flush=True)

@torch.no_grad()
def analyze_b(seed):
 torch.set_num_threads(2);dest=E27/f'B_s{seed}';done=json.loads((dest/'training_complete.json').read_text());model=Coefficients().cuda();model.load_state_dict(torch.load(dest/'selected.pt',map_location='cuda',weights_only=False)['model']);phi=load_phi(seed);bundle=bundle_for('C',seed);qs=torch.linspace(0,63,16,device='cuda').round().long();conds=[];info=[]
 for idx,path in enumerate(files('development')):
  d,s,c,zt,bt,lm=prepare(bundle,path);F=phi(d['q'],s);z,beta=model(s,F[qs],c['mass'][qs]);p=np.load(E25/f'C_s{seed}'/f'panel_{idx:04d}.npz');states=torch.as_tensor(p['states'],device='cuda');r=torch.as_tensor(p['remove'],device='cuda');a=torch.as_tensor(p['add'],device='cuda');norm=d['dense'][qs].square().sum()/16+1e-12;out={}
  for name,be in [('PredictedBeta',beta),('TrueBeta',bt[qs])]:
   ps=np.stack([safe_predictions(z,be,lm[qs],st,r,a,norm)[0].cpu().numpy() for st in states]);out[name]=ps;conds.append(dict(seed=seed,event=str(path.relative_to(OUT)),method=name,**conditional_metrics(p['true_gain'],ps)))
  info.append(dict(seed=seed,event=str(path.relative_to(OUT)),residual_nmse=nmse(z,zt[qs]),beta_nmse=nmse(beta,bt[qs]),max_abs_beta=float(beta.abs().max())))
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=p['event'],states=p['states'],remove=p['remove'],add=p['add'],true_gain=p['true_gain'],**out)
 dump(dest/'conditional.json',conds);dump(dest/'information.json',info);dump(dest/'analysis_complete.json',dict(seed=seed,events=288,test_accessed=False,selected_sha=sha(dest/'selected.pt')));print('B ANALYZED',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--analyze',action='store_true');args=p.parse_args();analyze_b(args.seed) if args.analyze else train_b(args.seed)
