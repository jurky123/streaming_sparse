"""E25-D: same reader trained from scratch on frozen C, no Test access."""
from .run import *
from ..e24_joint_representatives.train import evaluate

def train(seed):
 assert json.loads((E25/'D_gate.json').read_text())['passed']
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);dest=E25/f'D_s{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'training_complete.json').exists():return
 b=Bundle();source=torch.load(checkpoint('C',seed),map_location='cuda',weights_only=False);b.layers.load_state_dict(source['layers'])
 for p in b.layers.parameters():p.requires_grad_(False)
 # Reader initialized from scratch by Bundle, no original C reader weights loaded.
 original={n:t.detach().cpu().clone() for n,t in b.layers.state_dict().items()}
 f=torch.load(OUT/'fit.pt',weights_only=True);B=f['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long();trainfiles=files('train');devfiles=files('development');rng=np.random.default_rng(seed);opt=torch.optim.Adam(b.reader.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);curves=[];cks=[];start=time.time()
 torch.save(dict(reader=b.reader.state_dict(),seed=seed),dest/'initial_reader.pt')
 for epoch in range(4):
  for j,idx in enumerate(rng.permutation(len(trainfiles))):
   d=load(trainfiles[idx]);m=b.layers[d['key']]
   with torch.no_grad():
    s=b.make(d,'C');c=contributions(d['q'],s,m.basis,d['log_scale']);mask,r,a=route(d['q'][qs],s,m,d['own']);ff,_=RouteOps.features(d['q'][qs][None],s);risk=(ff*m.theta[:,None,None]).sum(-1).mean(1)[0].cpu().numpy();mask,r,a=getstate(c,d,mask,r,a,(epoch*len(trainfiles)+j)%10,risk,rng);zt,ut,bt,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a)
   cp={n:x[qs] for n,x in c.items()};z,u,beta=b.reader(d['q'][qs],s,cp);res=nn.functional.smooth_l1_loss(z/f['zscale'],zt[qs]/f['zscale'])+nn.functional.smooth_l1_loss(beta/f['bscale'],bt[qs]/f['bscale']);pg=predicted(z,u,cp['mass'],mask,r,a,beta)/(d['dense'].square().sum()/64);gs=f['gscale'];target=torch.softmax(torch.cat([g,g.new_zeros(1)])/(.3*gs),0);rank=-(target*torch.log_softmax(torch.cat([pg,pg.new_zeros(1)])/gs,0)).sum();ls=nn.functional.smooth_l1_loss(pg/gs,g/gs)+.1*rank;loss=res+ls;assert torch.isfinite(loss)
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(b.reader.parameters(),1,error_if_nonfinite=True);opt.step();curves.append(dict(epoch=epoch,update=epoch*len(trainfiles)+j,residual=float(res),set_loss=float(ls),total=float(loss)))
   if j%100==0:dump(dest/'training_status.json',dict(epoch=epoch,update=j,seconds=time.time()-start));print('D',seed,epoch,j,float(loss),flush=True)
  with torch.no_grad():rows=evaluate(b,'C',devfiles,B)
  old=sum(x['old'] for x in rows);initial=sum(x['initial'] for x in rows);final=sum(x['set'] for x in rows);ck=dict(epoch=epoch,reader={n:t.cpu() for n,t in b.reader.state_dict().items()},layers={n:t.cpu() for n,t in b.layers.state_dict().items()},initial_ratio=initial/old,final_ratio=final/old);torch.save(ck,dest/f'epoch{epoch}.pt');dump(dest/f'dev_epoch{epoch}.json',rows);cks.append(dict(epoch=epoch,initial_ratio=initial/old,final_ratio=final/old));dump(dest/'curves.json',curves);print('EPOCH',seed,cks[-1],flush=True)
 best=min(cks,key=lambda x:x['final_ratio']);ck=torch.load(dest/f'epoch{best["epoch"]}.pt',map_location='cpu',weights_only=False);torch.save(ck,dest/'selected.pt');assert all(torch.equal(original[n],t.detach().cpu()) for n,t in b.layers.state_dict().items());dump(dest/'training_complete.json',dict(seed=seed,updates=len(curves),epochs=4,seconds=time.time()-start,best=best,checkpoints=cks,source_C_sha=sha(checkpoint('C',seed)),frozen_tensors=len(original),frozen_exact=True,reader_initialized_from_scratch=True,test_accessed=False));print('TRAIN DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args();train(args.seed)
