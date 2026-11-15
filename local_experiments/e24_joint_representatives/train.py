from .core import *
from ..e12_swap_prediction.core import metric
from ..e19_response_moments.moments import gain_scores
from scipy.stats import spearmanr
import argparse,os

def files(split):
 cases=json.loads((OUT/'protocol.json').read_text())['cases'];ans=[]
 for c in cases:
  if c['split']!=split:continue
  p=OUT/'trajectories'/c['id']/'complete.json';assert p.exists(),p;ans.extend(OUT/x for x in json.loads(p.read_text())['files'])
 return sorted(ans)
def load(path):
 ev=torch.load(path,map_location='cpu',weights_only=True);kv=torch.load(path.parent/ev['kv_file'],map_location='cpu',weights_only=True);d={n:ev[n].cuda().float() if n=='q' else ev[n].cuda().double() for n in ['q','N','Z','dense','log_scale']};d['logZ']=(ev['page_logz']-ev['log_scale'][:,None]).cuda();d.update(ev=ev,k=kv['k'][None].cuda(),v=kv['v'][None].cuda(),hp=kv['history_pages'],old=Summary(*(kv['summary'][n].cuda() for n in ['keys','mean','detail','stats'])),own=ev['own'],key=f'l{ev["layer"]}h{ev["head"]}');return d
class Bundle(nn.Module):
 def __init__(self):
  super().__init__();state=torch.load(PRE,map_location='cpu',weights_only=False)['model'];self.layers=nn.ModuleDict({f'l{l}h{h}':layer(state,l,h,'cuda') for l in [1,15,29] for h in [0,3,6,9]});self.reader=Reader().cuda()
 def make(self,d,group):return d['old'] if group=='A' else summary(self.layers[d['key']],d['k'],d['v'],d['hp'])

def getstate(c,d,mask,r,a,mode,risk_scores=None,rng=None):
 if mode in [6,7]:
  for _ in range(1 if mode==6 else 2):
   g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a).detach();j=int(g.argmax())
   if g[j]>0:mask=mask.clone();mask[r[j]]=False;mask[a[j]]=True
   rr,aa=pool({'risk':risk_scores},mask.cpu().numpy(),d['own']);r=torch.as_tensor(rr,device='cuda');a=torch.as_tensor(aa,device='cuda')
 elif mode>=8:
  rs=torch.where(mask)[0];rs=rs[rs!=d['own']];aa=torch.where(~mask)[0];mask=mask.clone();mask[rs[int(rng.integers(len(rs)))]]=False;mask[aa[int(rng.integers(len(aa)))]]=True;rr,aa=pool({'risk':risk_scores},mask.cpu().numpy(),d['own']);r=torch.as_tensor(rr,device='cuda');a=torch.as_tensor(aa,device='cuda')
 return mask,r,a
@torch.no_grad()
def fit():
 import fcntl
 lock=(OUT/'fit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
 dest=OUT/'fit.pt'
 if dest.exists():return
 bundle=Bundle();cov=torch.zeros(128,128,device='cuda',dtype=torch.float64);beta2=[];gain2=[];names=[];data=files('train');qix=torch.linspace(0,63,16,device='cuda').round().long()
 for i,p in enumerate(data):
  d=load(p);m=bundle.layers[d['key']];c=contributions(d['q'],d['old'],m.basis,d['log_scale']);A=c['A'];dz=d['Z']-c['zh'];t=(d['N']-c['nh']-c['o'][:,None]*dz[...,None])/A[:,None,None];assert torch.isfinite(t).all(), str(p);v=t.reshape(-1,128).double();v=v/v.abs().max().clamp_min(1e-300);mat=v.T@v;cov+=mat/mat.trace().clamp_min(1e-30)/len(data);bs=d['logZ']-c['logzh'];beta2.append(float(bs.square().mean().sqrt()));mask,r,a=route(d['q'][qix],d['old'],m,d['own']);g=gains(c,d['N'],d['Z'],d['dense'],mask,r[::16],a[::16]);gain2.append(float(g.square().mean().sqrt()));names.append(str(p.relative_to(OUT)))
  if i%100==0:print('fit',i,flush=True)
 torch.save(cov.cpu(),OUT/'fit_covariance.pt');assert torch.isfinite(cov).all(), 'nonfinite training covariance';_,Unp=np.linalg.eigh((cov.cpu().numpy()+cov.cpu().numpy().T)/2);B=torch.from_numpy(Unp[:,-16:][:,::-1].copy()).float().cuda();z2=[]
 for p in data:
  d=load(p);m=bundle.layers[d['key']];c=contributions(d['q'],d['old'],m.basis,d['log_scale']);z,_,_,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);zm=z.abs().max();z2.append(float(zm*(z/zm.clamp_min(1e-300)).square().mean().sqrt()))
 torch.save(dict(B=B.cpu(),zscale=float(np.median(z2)),bscale=float(np.median(beta2)),gscale=max(float(np.median(gain2)),1e-8),train_files=names,scope='trainonly frozen original summary, uncentered equaltracePCA; fixed median of per-event RMS scales;32evenlyspacedMediumactions for gainRMS'),dest);print('fit complete',flush=True)
@torch.no_grad()
def evaluate(bundle,group,paths,B):
 rows=[];qs=torch.linspace(0,63,16,device='cuda').round().long()
 for p in paths:
  d=load(p);m=bundle.layers[d['key']];s=bundle.make(d,group);c=contributions(d['q'],s,m.basis,d['log_scale']);mask,r,a=route(d['q'][qs],s,m,d['own']);assert int(mask.sum())==max(1,math.floor(len(mask)*.1)) and mask[d['own']];J=objective(c,d['N'],d['Z'],d['dense'],mask);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);oldc=contributions(d['q'],d['old'],m.basis,d['log_scale']);oldmask=d['ev']['old_mask'].cuda();old=objective(oldc,d['N'],d['Z'],d['dense'],oldmask);oracle=g.clamp_min(0).max();row=dict(event=str(p.relative_to(OUT)),trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],old=float(old),initial=float(J),oracle_gain=float(oracle),set=float(J),independent=float(J),set_gain=0.,independent_gain=0.,execute=False)
  if group!='B':
   cp={n:x[qs] for n,x in c.items()};z,u,beta=bundle.reader(d['q'][qs],s,cp);score=predicted(z,u,cp['mass'],mask,r,a,beta);dzall=(cp['mass']*beta.to(u.dtype).exp()).sum(1);net=z-z.sum(1)[:,None]/dzall[:,None,None]*u[...,None];page=(net.square().sum(-1)/dzall[:,None].square()).mean(0);ind=page[a]-page[r]
   for name,scores in [('set',score),('independent',ind)]:
    j=int(scores.argmax());ex=bool(scores[j]>0);gain=float(g[j]) if ex else 0;row[name]=float(J)-gain;row[name+'_gain']=gain;row[name+'_remove']=int(r[j]);row[name+'_add']=int(a[j]);row[name+'_execute']=ex
    if name=='set':row.update(execute=ex,remove=int(r[j]),add=int(a[j]))
   targetz,targetu,targetb,t=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);row['residual_nmse']=float((z-targetz[qs]).square().sum()/(targetz[qs].square().sum()+1e-30));row['beta_nmse']=float((beta-targetb[qs]).square().sum()/(targetb[qs].square().sum()+1e-30));row['residual_cosine']=float(((z*targetz[qs]).sum(-1)/(z.norm(dim=-1)*targetz[qs].norm(dim=-1)+1e-30)).mean());row['rank_spearman']=float(spearmanr(score.cpu().numpy(),g.cpu().numpy()).statistic)
   truee=(d['N']-c['nh']-d['dense'][:,None]*(d['Z']-c['zh'])[...,None])/c['A'][:,None,None];trueZ=d['Z']/c['A'][:,None];prednet=z-z.sum(1)[:,None]/((cp['mass']*beta.to(u.dtype).exp()).sum(1))[:,None,None]*u[...,None]
   _,_,pi=gain_scores(prednet,cp['mass']+u,cp['mass'],mask,r,a,torch.arange(16,device='cuda'),True);_,_,ti=gain_scores(truee,trueZ,c['mass'],mask,r,a,qs,True);valid=abs(ti['I'])>1e-12;row['I_sign_accuracy']=float(np.mean((pi['I'][valid]>0)==(ti['I'][valid]>0))) if valid.any() else None;row['B_nmse']=float(np.sum((pi['B']-ti['B'])**2)/(np.sum(ti['B']**2)+1e-30));row['regret']=float(oracle)-row['set_gain']
   z64,u64,b64= bundle.reader(d['q'],s,c);s64=predicted(z64,u64,c['mass'],mask,r,a,b64);jj=int(s64.argmax());row['set64_gain']=float(g[jj]) if s64[jj]>0 else 0.

  rows.append(row)
 return rows

def run(group,seed,matched=False):
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);dest=OUT/'training'/f'{group+"_matched" if matched else group}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 bundle=Bundle();fitdata=torch.load(OUT/'fit.pt',weights_only=True);B=fitdata['B'].cuda();trainfiles=files('train');devfiles=files('development');qs=torch.linspace(0,63,16,device='cuda').round().long()
 if group=='C' or matched:
  warm=torch.load(OUT/'training'/f'A_s{seed}'/'selected.pt',weights_only=False,map_location='cuda');bundle.reader.load_state_dict(warm['reader'])
 for p in bundle.layers.parameters():
  # Layer trainability was initialized precisely; A freezes all.
  if group=='A':p.requires_grad_(False)
 for p in bundle.reader.parameters():p.requires_grad_(group!='B')
 opt=torch.optim.Adam([p for p in bundle.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);rng=np.random.default_rng(seed);curve=[];checkpoints=[];begin=time.time()
 for epoch in range(2):
  order=rng.permutation(len(trainfiles))
  for j,idx in enumerate(order):
   d=load(trainfiles[idx]);m=bundle.layers[d['key']];s=bundle.make(d,group);c=contributions(d['q'],s,m.basis,d['log_scale']);mask,r,a=route(d['q'][qs],s,m,d['own']);mode=(epoch*len(order)+j)%10
   with torch.no_grad():
    ff,_=RouteOps.features(d['q'][qs][None],s);risk=(ff*m.theta[:,None,None]).sum(-1).mean(1)[0].cpu().numpy();mask,r,a=getstate(c,d,mask,r,a,mode,risk,rng)
   approx=objective(c,d['N'],d['Z'],d['dense'],mask);res=approx.new_zeros(());ls=res
   if group!='B':
    cp={n:x[qs] for n,x in c.items()};z,u,beta=bundle.reader(d['q'][qs],s,cp);zt,ut,bt,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);res=nn.functional.smooth_l1_loss(z/fitdata['zscale'],zt[qs].detach()/fitdata['zscale'])+nn.functional.smooth_l1_loss(beta/fitdata['bscale'],bt[qs].detach()/fitdata['bscale']);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a).detach();pg=predicted(z,u,cp['mass'],mask,r,a,beta)/(d['dense'].square().sum()/64).detach();gs=fitdata['gscale'];target=torch.softmax(torch.cat([g,g.new_zeros(1)])/(.3*gs),0);rank=-(target*torch.log_softmax(torch.cat([pg,pg.new_zeros(1)])/gs,0)).sum();ls=nn.functional.smooth_l1_loss(pg/gs,g/gs)+.1*rank
   loss=res+ls if group=='A' else approx if group=='B' else approx+.1*res+.1*ls
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in bundle.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step();curve.append(dict(epoch=epoch,update=epoch*len(order)+j,approx=float(approx),residual=float(res),set_loss=float(ls),total=float(loss)))
   if j%100==0:dump(dest/'status.json',dict(status='training',epoch=epoch,update=j,seconds=time.time()-begin));print(group,seed,epoch,j,float(loss),flush=True)
  rows=evaluate(bundle,group,devfiles,B);old=sum(x['old'] for x in rows);initial=sum(x['initial'] for x in rows);final=sum(x['set'] for x in rows);ck=dict(epoch=epoch,reader={n:t.cpu() for n,t in bundle.reader.state_dict().items()},layers={n:t.cpu() for n,t in bundle.layers.state_dict().items()},initial_ratio=initial/old,final_ratio=final/old);torch.save(ck,dest/f'epoch{epoch}.pt');dump(dest/f'dev_epoch{epoch}.json',rows);checkpoints.append(dict(epoch=epoch,initial_ratio=initial/old,final_ratio=final/old))
  dump(dest/'curves.json',curve)
 valid=[x for x in checkpoints if x['initial_ratio']<=1.02];best=min(valid or checkpoints,key=lambda x:x['final_ratio']);chosen=torch.load(dest/f'epoch{best["epoch"]}.pt',weights_only=False);torch.save(chosen,dest/'selected.pt');dump(dest/'complete.json',dict(group=group,seed=seed,best=best,checkpoints=checkpoints,seconds=time.time()-begin,updates=len(curve)));print('DONE',group,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--matched',action='store_true');p.add_argument('--group',choices=['A','B','C']);p.add_argument('--seed',type=int,default=17);a=p.parse_args();fit() if a.fit else run(a.group,a.seed,a.matched)
