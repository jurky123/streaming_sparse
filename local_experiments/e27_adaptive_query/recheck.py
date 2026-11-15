from .analyze import *
import shutil
@torch.no_grad()
def recheck(seed):
 torch.set_num_threads(2);dest=E27/f's{seed}';backup=dest/'before_stable_gain';backup.mkdir(exist_ok=True);b=bundle_for('C',seed);meta=[]
 for ep in [-1,0,1,2,3]:
  name='initial' if ep==-1 else f'epoch{ep}';net=load_phi(seed,name);fname='development_initial.json' if ep==-1 else f'development_epoch{ep}.json'
  if not (backup/fname).exists():shutil.copy2(dest/fname,backup/fname)
  rows=evaluate(net,b,seed,files('development'));dump(dest/fname,rows);gain=sum(r['set_gain'] for r in rows);ora=sum(r['oracle_gain'] for r in rows);m=dict(epoch=ep,gain=gain,recovery=gain/ora,mean_loss=float(np.mean([x['reconstruction_loss'] for x in rows])));print('RECHECK',seed,m,flush=True)
  if ep>=0:meta.append(m)
 for name in ['training_complete.json','selected.pt']:
  if not (backup/name).exists():shutil.copy2(dest/name,backup/name)
 best=max(meta,key=lambda x:x['gain']);ck=torch.load(dest/f'epoch{best["epoch"]}.pt',map_location='cpu',weights_only=False);ck.update(best);torch.save(ck,dest/'selected.pt');done=json.loads((dest/'training_complete.json').read_text());done.update(best=best,checkpoints=meta,numerical_revision='Equivalent stable incremental gain; all five preexisting checkpoints reevaluated, no training repeated');dump(dest/'training_complete.json',done)
 for name in ['analysis_complete.json','decisions.json','conditional.json','fit_quality.json']:
  if (dest/name).exists():shutil.move(dest/name,backup/name)
 analyze(seed)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args();recheck(args.seed)
