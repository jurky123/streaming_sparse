from .core import *

def train(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);dest=E27/f's{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'training_complete.json').exists():return
 bundle=bundle_for('C',seed)
 for p in bundle.parameters():p.requires_grad_(False)
 original={k:v.detach().cpu().clone() for k,v in bundle.layers.state_dict().items()}
 net=AdaptivePhi().cuda();fit=torch.load(OUT/'fit.pt',weights_only=True);trainfiles=files('train');devfiles=files('development');rng=np.random.default_rng(seed);opt=torch.optim.Adam(net.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);start=time.time();curves=[];checkpoints=[]
 torch.save(dict(model=net.state_dict(),epoch=-1),dest/'initial.pt');rows=evaluate(net,bundle,seed,devfiles);dump(dest/'development_initial.json',rows);print('INITIAL',seed,sum(x['set_gain'] for x in rows)/sum(x['oracle_gain'] for x in rows),flush=True)
 for epoch in range(4):
  for j,idx in enumerate(rng.permutation(len(trainfiles))):
   d,s,c,zt,bt,lm=prepare(bundle,trainfiles[idx]);F=net(d['q'],s);loss=variable_projection_loss(F,lm,zt,fit['zscale']);assert torch.isfinite(loss)
   opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step();curves.append(dict(epoch=epoch,update=epoch*len(trainfiles)+j,loss=float(loss),gradient_norm=float(grad)))
   if j%100==0:dump(dest/'status.json',dict(epoch=epoch,update=j,seconds=time.time()-start));print(seed,epoch,j,float(loss),flush=True)
  rows=evaluate(net,bundle,seed,devfiles);gain=sum(x['set_gain'] for x in rows);oracle=sum(x['oracle_gain'] for x in rows);meta=dict(epoch=epoch,gain=gain,recovery=gain/oracle,mean_loss=float(np.mean([x['reconstruction_loss'] for x in rows])));checkpoints.append(meta);torch.save(dict(model=net.state_dict(),**meta),dest/f'epoch{epoch}.pt');dump(dest/f'development_epoch{epoch}.json',rows);dump(dest/'curves.json',curves);print('EPOCH',seed,meta,flush=True)
 best=max(checkpoints,key=lambda x:x['gain']);torch.save(torch.load(dest/f'epoch{best["epoch"]}.pt',map_location='cpu',weights_only=False),dest/'selected.pt');assert all(torch.equal(original[k],v.cpu()) for k,v in bundle.layers.state_dict().items());dump(dest/'training_complete.json',dict(seed=seed,updates=len(curves),best=best,checkpoints=checkpoints,seconds=time.time()-start,frozen_C_exact=True,C_sha=sha(checkpoint('C',seed)),fit_sha=sha(OUT/'fit.pt'),test_accessed=False,parameters=sum(p.numel() for p in net.parameters())));print('DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args();train(args.seed)
