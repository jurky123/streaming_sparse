from .core import *
@torch.no_grad()
def run(seed):
 torch.set_num_threads(2);b=bundle_for('C',seed);fit=torch.load(OUT/'fit.pt',weights_only=True);qs=torch.linspace(0,63,16,device='cuda').round().long();rows=[]
 for idx,path in enumerate(files('development')):
  d,s,c,tz,tb,lm=prepare(b,path);F=features(b.reader,d['q'],c);zc,_,_,_=fit_coeff(F,lm,tz,tb,fit['zscale'],fit['bscale']);zs,_=svd_rank5(tz);mask,r,a=route(d['q'][qs],s,b.layers[d['key']],d['own']);p=np.load(E25/f'C_s{seed}'/f'panel_{idx:04d}.npz');g=torch.as_tensor(p['true_initial_gain'],device='cuda');J=float(objective(c,d['N'],d['Z'],d['dense'],mask));norm=d['dense'][qs].square().sum()/16+1e-12
  for name,z in [('Cphi_trueBeta',zc),('SVD5_trueBeta',zs)]:
   sc,ind=safe_predictions(z[qs],tb[qs],lm[qs],mask,r,a,norm)
   for policy,v in [('Set',sc),('Independent',ind)]:
    j=int(v.argmax());ex=bool(v[j]>0);true=float(g[j]);rows.append(dict(seed=seed,event=str(path.relative_to(OUT)),trajectory=d['ev']['trajectory'],layer=d['ev']['layer'],chunk=d['ev']['chunk'],step=d['ev']['step'],head=d['ev']['head'],method=name,policy=policy,initial=J,oracle_gain=max(0.,float(g.max())),proposal_gain=true,hindsight_gain=max(0.,true),executed_gain=true if ex else 0.,execute=ex,remove=int(r[j]),add=int(a[j]),predicted_gain=float(v[j])))
 dump(E27/f'baselines_stable_s{seed}.json',rows);print('BASELINES',seed)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args();run(args.seed)
