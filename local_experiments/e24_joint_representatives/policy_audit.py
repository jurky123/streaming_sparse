from .train import *
import argparse
@torch.no_grad()
def run(group,seed):
 torch.set_num_threads(2);dest=OUT/'training'/f'{"A_matched" if group=="A" else group}_s{seed}';ck=torch.load(dest/'selected.pt',map_location='cuda',weights_only=False);bundle=Bundle();bundle.layers.load_state_dict(ck['layers']);bundle.reader.load_state_dict(ck['reader']);rows=json.loads((dest/f'dev_epoch{ck["epoch"]}.json').read_text());qs=torch.linspace(0,63,16,device='cuda').round().long();bmax=0.
 for row in rows:
  d=load(OUT/row['event']);m=bundle.layers[d['key']];s=bundle.make(d,group);c=contributions(d['q'],s,m.basis,d['log_scale']);mask,r,a=route(d['q'][qs],s,m,d['own']);cp={n:x[qs] for n,x in c.items()};z,u,b=bundle.reader(d['q'][qs],s,cp);sc=predicted(z,u,cp['mass'],mask,r,a,b);assert torch.isfinite(sc).all();j=int(sc.argmax());assert bool(sc[j]>0)==row['set_execute'] and int(r[j])==row['set_remove'] and int(a[j])==row['set_add'];bmax=max(bmax,float(b.abs().max()))
 dump(OUT/'audit_models'/f'policy_{group}_s{seed}.json',dict(passed=True,events=len(rows),all512scores_finite=True,decisions_reproduced=True,max_abs_predicted_beta=bmax,checkpoint_sha=sha(dest/'selected.pt')));print('policy',group,seed,bmax)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group');p.add_argument('--seed',type=int);a=p.parse_args();run(a.group,a.seed)
