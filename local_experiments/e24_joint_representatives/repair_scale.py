from .core import *
from .train import files
import argparse
@torch.no_grad()
def run(worker,workers):
 torch.set_num_threads(2);paths=files('train')+files('development')+files('test');checks=[]
 for ii,p in enumerate(paths[worker::workers]):
  x=torch.load(p,weights_only=True);kv=torch.load(p.parent/x['kv_file'],weights_only=True)
  if x.get('scale_version')=='page_log_center_v3':continue
  q=x['q'].cuda().double();k=kv['k'].cuda().double();v=kv['v'].cuda().double();raw=torch.einsum('qd,ptd->qpt',q,k)/math.sqrt(128);lz=raw.logsumexp(-1);op=torch.einsum('qpt,ptd->qpd',raw.softmax(-1),v);keys=kv['summary']['keys'][0].cuda().double();ls=(torch.einsum('qd,prd->qpr',q,keys)/math.sqrt(128)+math.log(16)).logsumexp(-1);low=torch.minimum(lz.amin(1),ls.amin(1));high=torch.maximum(lz.amax(1),ls.amax(1));scale=high-300;Z=(lz-scale[:,None]).exp();N=op*Z[...,None];od=(lz.softmax(1)[...,None]*op).sum(1);assert torch.isfinite(N).all() and (Z.sum(1)>0).all() and torch.isfinite(lz).all();checks.append(dict(path=str(p.relative_to(OUT)),old_sha=sha(p),max_log_range=float((high-low).max()),previous_zero_pages=int((x['Z']==0).sum()),dense_change=float((od.cpu()-x['dense']).abs().max())))
  x.update(N=N.cpu(),Z=Z.cpu(),dense=od.cpu(),log_scale=scale.cpu(),page_logz=lz.cpu(),page_output=op.cpu(),scale_version='page_log_center_v3');tmp=p.with_suffix('.repair');torch.save(x,tmp);tmp.replace(p)
  if ii%100==0:print(worker,ii,flush=True)
 dump(OUT/f'scale_repair_worker{worker}.json',checks);print('repaired',worker,len(checks))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--worker',type=int);a.add_argument('--workers',type=int,default=8);v=a.parse_args();run(v.worker,v.workers)
