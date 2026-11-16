"""E39 fixed architecture continuation; train-only hard competitors, sealed test."""
from ..e38_action_attribution.core import *
E38=DEST
DEST=ROOT/'artifacts/e39_action_margin_v1'

def margin_loss(pg,g,scale):
 p=torch.cat([pg,pg.new_zeros(1)]);t=torch.cat([g,g.new_zeros(1)])
 best=int(t.argmax()) if float(g.max())>0 else len(g)
 hard=torch.argsort(p.detach(),descending=True,stable=True)[:4]
 H=torch.unique(torch.cat([hard,hard.new_tensor([best,len(g)])]),sorted=True)
 terms=(p[H]-p[best]+t[best]-t[H])/scale
 return terms.max(),H,best

def original_set_loss(pg,g,gs):
 target=torch.softmax(torch.cat([g,g.new_zeros(1)])/(.3*gs),0)
 rank=-(target*torch.log_softmax(torch.cat([pg,pg.new_zeros(1)])/gs,0)).sum()
 return nn.functional.smooth_l1_loss(pg/gs,g/gs)+.1*rank

@torch.no_grad()
def decomposition(seed):
 torch.set_num_threads(2);qs=torch.linspace(0,63,16,device='cuda').round().long();rows=[]
 old=json.loads((E38/f'B_s{seed}/action_margin_attribution.json').read_text())
 for i,path in enumerate(files('development')):
  x=torch.load(E30/f's{seed}/beta_{i:04d}.pt',map_location='cuda',weights_only=True);t=torch.load(E38/f'B_s{seed}/target_{i:04d}.pt',map_location='cuda',weights_only=True);p=np.load(E30/f's{seed}/panel_{i:04d}.npz');o=old[i];assert x['event']==o['event']==t['event']==str(p['event'])
  mask=torch.as_tensor(p['states'][0],device='cuda');r=torch.as_tensor(p['initial_remove'],device='cuda');a=torch.as_tensor(p['initial_add'],device='cuda');pair=[o['j_pred'],o['j_oracle']];rt=t['true_response'].double();bt=x['beta_true'].double();lm=x['logmass'].double();cen=x['center'].double();norm=t['norm64']
  v16=pair_scores(rt[qs],bt[qs],lm[qs],cen[qs],mask,r,a,pair,norm);v64=pair_scores(rt,bt,lm,cen,mask,r,a,pair,norm)
  m16=v16[0]-v16[1];m64=v64[0]-v64[1];mf=o['M_true'];mp=o['M00'];assert abs(m16-o['M11'])<1e-8
  err=[mp-m16,m16-m64,m64-mf];closure=abs(sum(err)-(mp-mf));assert closure<1e-10
  rows.append(dict(**t['meta'],j_pred=pair[0],j_oracle=pair[1],same_action=pair[0]==pair[1],M_pred=mp,M_true16=m16,M_true64_B16=m64,M_true64_full=mf,prediction_error=err[0],query_difference=err[1],projection_difference=err[2],closure=closure,true16_correct=m16<=1e-12,true64_B16_correct=m64<=1e-12,original_harmful=o['proposed_gain']< -1e-12))
 dump(DEST/f'decomposition_s{seed}.json',rows);print('DECOMPOSITION DONE',seed,len(rows),flush=True)

@torch.no_grad()
def evaluate_all(models,phi,bundle,seed,epoch,split='development'):
 qs=torch.linspace(0,63,16,device='cuda').round().long();result={name:[] for name in models};conds={name:[] for name in models};diag={name:[] for name in models}
 paths=files(split)
 if split=='train':paths=paths[::24] # predetermined 48 training events, no extra optimization
 for idx,path in enumerate(paths):
  d,s,c,rt,bt,lm,cen=prepare_local(bundle,path);F=phi(d['q'],s)[qs];mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);norm=d['dense'].square().sum()/64+1e-12/64;J=float(objective(c,d['N'],d['Z'],d['dense'],mask));meta=dict(seed=seed,event=str(path.relative_to(OUT)),**{k:d['ev'][k] for k in ['trajectory','layer','chunk','step','head','tile']});ora=max(0.,float(g.max()));jo=pick(g)
  if split=='development':
   panel=np.load(E30/f's{seed}/panel_{idx:04d}.npz');assert np.array_equal(mask.cpu(),panel['states'][0]) and np.array_equal(r.cpu(),panel['initial_remove']) and np.array_equal(a.cpu(),panel['initial_add']);assert np.allclose(g.cpu(),panel['true_initial_gain'],atol=1e-8,rtol=1e-6);old=pick(torch.as_tensor(panel['initial_Predicted']));oldgain=value(g,old)
  for name,model in models.items():
   rp,bp=model(s,F);sc,ind=local_scores(rp,bp,lm[qs],cen[qs],mask,r,a,norm);order=torch.argsort(sc,descending=True,stable=True);ml,hh,jj=margin_loss(sc,g,rms_floor(g,(0,),torch.load(OUT/'fit.pt',weights_only=True)['gscale']).reshape(()))
   for policy,scores in [('Set',sc),('Independent',ind)]:
    j=pick(scores);gain=value(g,j);prop=int(scores.argmax());result[name].append(dict(**meta,group=name,epoch=epoch,policy=policy,initial=J,oracle_gain=ora,executed_gain=gain,execute=j>=0,action=j,proposal_gain=value(g,prop),hindsight_gain=max(0.,value(g,prop)),top1=max(0.,value(g,int(order[0]))),top4=max(0.,float(g[order[:4]].max())),top8=max(0.,float(g[order[:8]].max())),predicted_gain=value(scores,j),fixed_margin=value(sc,old)-value(sc,jo) if split=='development' else None,true_fixed_margin=oldgain-value(g,jo) if split=='development' else None,old_harmful=oldgain< -1e-12 if split=='development' else None,old_action_rejected=value(sc,old)<=0 if split=='development' else None))
   diag[name].append(dict(**meta,margin_loss=float(ml),local_nmse=nmse(rp,rt[qs]),beta_nmse=nmse(bp,bt[qs])))
   if split=='development':
    rr=torch.as_tensor(panel['remove'],device='cuda');aa=torch.as_tensor(panel['add'],device='cuda');ps=np.stack([local_scores(rp,bp,lm[qs],cen[qs],torch.as_tensor(st,device='cuda'),rr,aa,norm)[0].cpu().numpy() for st in panel['states']]);conds[name].append(dict(**meta,**conditional_metrics(panel['true_gain'],ps)))
    dest=DEST/f'{name}_s{seed}';np.savez_compressed(dest/f'epoch{epoch}_panel_{idx:04d}.npz',event=meta['event'],scores=sc.cpu().numpy(),independent=ind.cpu().numpy(),true_gain=g.cpu().numpy(),conditional=ps,conditional_true=panel['true_gain'],mask=mask.cpu().numpy(),remove=r.cpu().numpy(),add=a.cpu().numpy())
 for name in models:
  dest=DEST/f'{name}_s{seed}';dump(dest/f'{split}_epoch{epoch}.json',result[name]);dump(dest/f'{split}_information_epoch{epoch}.json',diag[name]);dump(dest/f'{split}_conditional_epoch{epoch}.json',conds[name])
 print('EVALUATED',seed,epoch,split,flush=True)

def train(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);bundle=bundle_for('C',seed);phi=load_phi(seed)
 for p in itertools.chain(bundle.parameters(),phi.parameters()):p.requires_grad_(False)
 frozen={n:t.cpu().clone() for n,t in bundle.layers.state_dict().items()};fphi={n:t.cpu().clone() for n,t in phi.state_dict().items()};init=torch.load(E29/f's{seed}/selected.pt',map_location='cuda',weights_only=False)['model'];models={};opts={};curves={}
 for name in ['T0','T1']:
  dest=DEST/f'{name}_s{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'training_complete.json').exists();m=LocalReader().cuda();m.load_state_dict(init);models[name]=m;opts[name]=torch.optim.Adam(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);curves[name]=[];torch.save(m.state_dict(),dest/'initial.pt')
 assert all(torch.equal(models['T0'].state_dict()[k],v) for k,v in models['T1'].state_dict().items())
 scale=json.loads((E29/f's{seed}/local_scale.json').read_text())['scale'];fit=torch.load(OUT/'fit.pt',weights_only=True);qs=torch.linspace(0,63,16,device='cuda').round().long();rng=np.random.default_rng(seed);paths=files('train');assert len(paths)==1152;start=time.time();schedule=[]
 evaluate_all({'E29':models['T0']},phi,bundle,seed,-1,'train') if False else None
 # Matched frozen 48-event train panel baseline, saved in each group.
 evaluate_all(models,phi,bundle,seed,-1,'train')
 for epoch in range(2):
  for pos,ix in enumerate(rng.permutation(len(paths))):
   update=epoch*len(paths)+pos;d,s,c,rt,bt,lm,cen=prepare_local(bundle,paths[ix])
   with torch.no_grad():
    F=phi(d['q'],s);mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);ff,_=RouteOps.features(d['q'][qs][None],s);risk=(ff*bundle.layers[d['key']].theta[:,None,None]).sum(-1).mean(1)[0].cpu().numpy();mask,r,a=getstate(c,d,mask,r,a,update%10,risk,rng);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);bs=rms_floor(bt[qs],(0,),fit['bscale']);gs=rms_floor(g,(0,),fit['gscale']).reshape(())
   schedule.append(dict(update=update,event=str(paths[ix].relative_to(OUT)),mode=update%10,mask=mask.cpu().tolist(),remove=r.cpu().tolist(),add=a.cpu().tolist()))
   for name,model in models.items():
    rp,beta=model(s,F[qs]);local=((c['mass'][qs][...,None]*(rp-rt[qs]))/scale).square().mean();quality=nn.functional.smooth_l1_loss(beta/bs,bt[qs]/bs)
    # Preserve E29 training arithmetic for both arms; no scoring change confound.
    z=reconstruct(rp,beta,lm[qs],cen[qs]);pg,_=safe_predictions(z,beta,lm[qs],mask,r,a,d['dense'].square().sum()/64+1e-12)
    ml,H,best=margin_loss(pg,g,gs);sl=original_set_loss(pg,g,gs) if name=='T0' else ml;loss=local+quality+sl;assert torch.isfinite(loss);opt=opts[name];opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
    curves[name].append(dict(epoch=epoch,update=update,event=str(paths[ix].relative_to(OUT)),local_loss=float(local),quality_loss=float(quality),set_loss=float(sl),margin_loss=float(ml),loss=float(loss),gradnorm=float(grad),gain_scale=float(gs),hard=H.tolist(),best=best))
   if pos%192==0:print('TRAIN',seed,epoch,pos,'seconds',round(time.time()-start),flush=True)
  for name,m in models.items():
   dest=DEST/f'{name}_s{seed}';torch.save(dict(model=m.state_dict(),optimizer=opts[name].state_dict(),epoch=epoch,updates=(epoch+1)*len(paths)),dest/f'epoch{epoch}.pt');dump(dest/'curves.json',curves[name])
  evaluate_all(models,phi,bundle,seed,epoch);evaluate_all(models,phi,bundle,seed,epoch,'train')
 assert all(torch.equal(v.cpu(),frozen[n]) for n,v in bundle.layers.state_dict().items());assert all(torch.equal(v.cpu(),fphi[n]) for n,v in phi.state_dict().items())
 dump(DEST/f'schedule_s{seed}.json',schedule)
 for name in models:
  dest=DEST/f'{name}_s{seed}';dump(dest/'training_complete.json',dict(seed=seed,group=name,updates=2304,final_epoch=1,checkpoint_selection='predeclared final, no development selection',parameters=sum(p.numel() for p in models[name].parameters()),schedule_sha=sha(DEST/f'schedule_s{seed}.json'),initial_sha=sha(E29/f's{seed}/selected.pt'),C_sha=sha(checkpoint('C',seed)),phi_sha=sha(E27/f's{seed}/selected.pt'),frozen_C_exact=True,frozen_phi_exact=True,test_accessed=False,seconds=time.time()-start))
 print('DONE TRAIN',seed,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',choices=['A','train'],required=True);args=p.parse_args();{'A':decomposition,'train':train}[args.mode](args.seed)
