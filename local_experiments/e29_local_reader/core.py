"""Fixed-capacity local response readout and analytic global centering."""
from ..e27_adaptive_query.coefficients import *
from ..e28_local_response.run import summary_output,reconstruct
E29=ROOT/'artifacts/e29_local_reader_v1'

class LocalReader(Coefficients):
 def forward(self,s,F,mass=None):
  co=self.page(page_features(s)).reshape(-1,5,17).bfloat16().float()
  val=co[None,:,0]+torch.einsum('qpj,pjk->qpk',F[...,1:].float(),co[:,1:])
  return val[...,:16],val[...,16]

@torch.no_grad()
def prepare_local(bundle,path):
 d,s,c,zt,bt,lm=prepare(bundle,path);B=torch.load(OUT/'fit.pt',weights_only=True)['B'].cuda().double();ap=summary_output(c,s,bundle.layers[d['key']].basis)
 rt=(d['ev']['page_output'].cuda().double()-ap)@B;center=(ap-c['o'][:,None])@B
 return d,s,c,rt,bt,lm,center

@torch.no_grad()
def fit_scale(seed):
 torch.set_num_threads(2);dest=E29/f's{seed}';dest.mkdir(exist_ok=True);bundle=bundle_for('C',seed);rows=[]
 for i,p in enumerate(files('train')):
  d,s,c,r,bt,lm,center=prepare_local(bundle,p);x=c['mass'][...,None]*r;v=x.abs().max();rms=float(v*(x/v.clamp_min(1e-300)).square().mean().sqrt());rows.append(dict(event=str(p.relative_to(OUT)),weighted_local_rms=rms))
  if i%192==0:print('SCALE',seed,i,flush=True)
 scale=float(np.median([r['weighted_local_rms'] for r in rows]));assert scale>0 and math.isfinite(scale)
 dump(dest/'local_scale.json',dict(seed=seed,scale=scale,events=rows,split='train',rule='median per-event RMS of mass*true localB16 response across all64Queries/pages/16coords',test_accessed=False));print('SCALE DONE',seed,scale,flush=True)

@torch.no_grad()
def evaluate(model,phi,bundle,seed,paths):
 qs=torch.linspace(0,63,16,device='cuda').round().long();rows=[]
 for path in paths:
  d,s,c,rt,bt,lm,center=prepare_local(bundle,path);F=phi(d['q'],s);mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);J=float(objective(c,d['N'],d['Z'],d['dense'],mask));rp,bp=model(s,F[qs]);norm=d['dense'][qs].square().sum()/16+1e-12
  for name,rr,bb in [('Local-Online',rp,bp),('Local-TrueBeta',rp,bt[qs]),('Local-TrueResponse',rt[qs],bp)]:
   z=reconstruct(rr,bb,lm[qs],center[qs]);sc,ind=safe_predictions(z,bb,lm[qs],mask,r,a,norm)
   for policy,score in [('Set',sc),('Independent',ind)]:
    j=int(score.argmax());ex=bool(score[j]>0);rows.append(dict(seed=seed,event=str(path.relative_to(OUT)),trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],tile=d['ev']['tile'],method=name,policy=policy,initial=J,oracle_gain=max(0.,float(g.max())),proposal_gain=float(g[j]),hindsight_gain=max(0.,float(g[j])),executed_gain=float(g[j]) if ex else 0.,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(score[j])))
 return rows

def train(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);dest=E29/f's{seed}';dest.mkdir(exist_ok=True)
 assert (E29/'teacher'/f's{seed}'/'complete.json').exists(), 'E29-A first'
 if (dest/'training_complete.json').exists():return
 scale=json.loads((dest/'local_scale.json').read_text())['scale'];bundle=bundle_for('C',seed);phi=load_phi(seed)
 for p in itertools.chain(bundle.parameters(),phi.parameters()):p.requires_grad_(False)
 frozen={n:t.cpu().clone() for n,t in bundle.layers.state_dict().items()};frozen_phi={n:t.cpu().clone() for n,t in phi.state_dict().items()}
 model=LocalReader().cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);fit=torch.load(OUT/'fit.pt',weights_only=True);qs=torch.linspace(0,63,16,device='cuda').round().long();rng=np.random.default_rng(seed);trainfiles=files('train');devfiles=files('development');curves=[];cks=[];start=time.time();torch.save(model.state_dict(),dest/'initial.pt')
 assert all(torch.equal(v.cpu(),torch.load(E27/f'B_s{seed}'/'initial.pt',map_location='cpu',weights_only=True)[n]) for n,v in model.state_dict().items()),'match initial weights'
 for epoch in range(4):
  for j,idx in enumerate(rng.permutation(len(trainfiles))):
   d,s,c,rt,bt,lm,center=prepare_local(bundle,trainfiles[idx])
   with torch.no_grad():
    F=phi(d['q'],s);mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);ff,_=RouteOps.features(d['q'][qs][None],s);risk=(ff*bundle.layers[d['key']].theta[:,None,None]).sum(-1).mean(1)[0].cpu().numpy();mask,r,a=getstate(c,d,mask,r,a,(epoch*len(trainfiles)+j)%10,risk,rng);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);bs=rms_floor(bt[qs],(0,),fit['bscale']);gs=rms_floor(g,(0,),fit['gscale']).reshape(())
   rp,beta=model(s,F[qs]);local=((c['mass'][qs][...,None]*(rp-rt[qs]))/scale).square().mean();quality=nn.functional.smooth_l1_loss(beta/bs,bt[qs]/bs);z=reconstruct(rp,beta,lm[qs],center[qs]);pg,_=safe_predictions(z,beta,lm[qs],mask,r,a,d['dense'].square().sum()/64+1e-12);target=torch.softmax(torch.cat([g,g.new_zeros(1)])/(.3*gs),0);rank=-(target*torch.log_softmax(torch.cat([pg,pg.new_zeros(1)])/gs,0)).sum();setloss=nn.functional.smooth_l1_loss(pg/gs,g/gs)+.1*rank;loss=local+quality+setloss
   assert torch.isfinite(loss),(seed,epoch,j,'nonfinite loss');opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();curves.append(dict(epoch=epoch,update=epoch*len(trainfiles)+j,event=str(trainfiles[idx].relative_to(OUT)),local_loss=float(local),quality_loss=float(quality),set_loss=float(setloss),loss=float(loss),gradnorm=float(grad),gain_scale=float(gs)))
   if j%100==0:dump(dest/'status.json',dict(epoch=epoch,update=j,seconds=time.time()-start));print('TRAIN',seed,epoch,j,float(loss),flush=True)
  rows=evaluate(model,phi,bundle,seed,devfiles);rr=[r for r in rows if r['method']=='Local-Online' and r['policy']=='Set'];gain=sum(r['executed_gain'] for r in rr);ora=sum(r['oracle_gain'] for r in rr);meta=dict(epoch=epoch,gain=gain,recovery=gain/ora);cks.append(meta);torch.save(dict(model=model.state_dict(),**meta),dest/f'epoch{epoch}.pt');dump(dest/f'development_epoch{epoch}.json',rows);dump(dest/'curves.json',curves);print('EPOCH',seed,meta,flush=True)
 best=max(cks,key=lambda x:x['gain']);torch.save(torch.load(dest/f'epoch{best["epoch"]}.pt',map_location='cpu',weights_only=False),dest/'selected.pt');assert all(torch.equal(v.cpu(),frozen[n]) for n,v in bundle.layers.state_dict().items());assert all(torch.equal(v.cpu(),frozen_phi[n]) for n,v in phi.state_dict().items());dump(dest/'training_complete.json',dict(seed=seed,updates=len(curves),best=best,checkpoints=cks,seconds=time.time()-start,frozen_C_exact=True,frozen_phi_exact=True,initial_matches_E27B=True,C_sha=sha(checkpoint('C',seed)),phi_sha=sha(E27/f's{seed}'/'selected.pt'),parameters=sum(p.numel() for p in model.parameters()),test_accessed=False));print('DONE',seed,flush=True)

@torch.no_grad()
def analyze(seed):
 torch.set_num_threads(2);dest=E29/f's{seed}';done=json.loads((dest/'training_complete.json').read_text());model=LocalReader().cuda();model.load_state_dict(torch.load(dest/'selected.pt',map_location='cuda',weights_only=False)['model']);phi=load_phi(seed);bundle=bundle_for('C',seed);qs=torch.linspace(0,63,16,device='cuda').round().long();conds=[];info=[]
 for idx,path in enumerate(files('development')):
  d,s,c,rt,bt,lm,center=prepare_local(bundle,path);F=phi(d['q'],s);rp,bp=model(s,F[qs]);panel=np.load(E25/f'C_s{seed}'/f'panel_{idx:04d}.npz');states=torch.as_tensor(panel['states'],device='cuda');r=torch.as_tensor(panel['remove'],device='cuda');a=torch.as_tensor(panel['add'],device='cuda');norm=d['dense'][qs].square().sum()/16+1e-12;out={}
  for name,rr,bb in [('Local-Online',rp,bp),('Local-TrueBeta',rp,bt[qs]),('Local-TrueResponse',rt[qs],bp)]:
   z=reconstruct(rr,bb,lm[qs],center[qs]);ps=np.stack([safe_predictions(z,bb,lm[qs],st,r,a,norm)[0].cpu().numpy() for st in states]);out[name]=ps;conds.append(dict(seed=seed,event=str(path.relative_to(OUT)),method=name,**conditional_metrics(panel['true_gain'],ps)))
  info.append(dict(seed=seed,event=str(path.relative_to(OUT)),local_nmse=nmse(rp,rt[qs]),mass_weighted_local_nmse=nmse(c['mass'][qs][...,None]*rp,c['mass'][qs][...,None]*rt[qs]),beta_nmse=nmse(bp,bt[qs]),max_abs_beta=float(bp.abs().max())))
  np.savez_compressed(dest/f'panel_{idx:04d}.npz',event=panel['event'],states=panel['states'],remove=panel['remove'],add=panel['add'],true_gain=panel['true_gain'],**out)
 dump(dest/'conditional.json',conds);dump(dest/'information.json',info);dump(dest/'analysis_complete.json',dict(seed=seed,events=288,test_accessed=False,selected_sha=sha(dest/'selected.pt')));print('ANALYZED',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['scale','train','analyze'],required=True);args=p.parse_args();{'scale':fit_scale,'train':train,'analyze':analyze}[args.mode](args.seed)
