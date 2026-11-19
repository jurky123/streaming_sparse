"""Resumable full-coverage E24-C joint training, without old subset fallbacks."""
import argparse,collections,json,math,time
import numpy as np
import torch
from torch import nn
from .common import *
from ..e24_joint_representatives.core import layer,Reader,summary,contributions,labels,route,gains,objective,predicted
from ..e24_joint_representatives.train import getstate
from ..review_v1.ops import Summary,RouteOps
from ..learned_cache.io import RunLock

class Bundle(nn.Module):
    def __init__(self,seed):
        super().__init__();state=torch.load(PRE,map_location='cpu',weights_only=False)['model']
        self.layers=nn.ModuleDict({f'l{l}h{h}':layer(state,l,h,'cuda') for l in LAYERS for h in HEADS})
        self.reader=Reader().cuda()
        self.reader.load_state_dict(torch.load(E24/f'training/A_s{seed}/selected.pt',map_location='cuda',weights_only=False)['reader'])
        self.frozen={k:v.detach().cpu().clone() for k,v in self.layers.state_dict().items() if k.rsplit('.',1)[-1] not in ('pool_k','pool_v','pool_feature')}
    def check_frozen(self):
        assert all(torch.equal(self.layers.state_dict()[k].cpu(),v) for k,v in self.frozen.items())

def load(path):
    ev=torch.load(path,map_location='cpu',weights_only=True)
    if ev['split'] not in ('train','development'):raise ValueError('Test sealed')
    kv=torch.load(path.parent/ev['kv_file'],map_location='cpu',weights_only=True)
    d={n:ev[n].cuda().float() if n=='q' else ev[n].cuda().double() for n in ('q','N','Z','dense','log_scale')}
    d.update(ev=ev,k=kv['k'][None].cuda(),v=kv['v'][None].cuda(),hp=kv['history_pages'],
             old=Summary(*(kv['summary'][n].cuda() for n in NAMES)),own=ev['own'],key=f'l{ev["layer"]}h{ev["head"]}')
    d['logZ']=ev['page_logz'].cuda().double()-d['log_scale'][:,None]
    return d

@torch.no_grad()
def evaluate(bundle,paths):
    rows=[];coverage=collections.Counter();qs=torch.linspace(0,63,16,device='cuda').round().long()
    for path in paths:
        d=load(path);m=bundle.layers[d['key']];s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale'])
        mask,r,a=route(d['q'][qs],s,m,d['own']);cp={n:v[qs] for n,v in c.items()};z,u,beta=bundle.reader(d['q'][qs],s,cp)
        score=predicted(z,u,cp['mass'],mask,r,a,beta);j=int(score.argmax());gain=gains(c,d['N'],d['Z'],d['dense'],mask,r[j:j+1],a[j:j+1])[0]
        oc=contributions(d['q'],d['old'],m.basis,d['log_scale']);om,_,_=route(d['q'][qs],d['old'],m,d['own'])
        rows.append(dict(event=str(path.relative_to(OUT)),key=d['key'],trajectory=d['ev']['trajectory'],old=float(objective(oc,d['N'],d['Z'],d['dense'],om)),initial=float(objective(c,d['N'],d['Z'],d['dense'],mask)),executed_gain=float(gain) if score[j]>0 else 0.))
        coverage[d['key']]+=1
    assert len(coverage)==348 and set(coverage.values())=={24},dict(coverage)
    return rows

def train(seed,limit=0):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);protocol()
    root=OUT/'training'/f'C_s{seed}';lock=RunLock(root);begin=time.perf_counter()
    try:
        paths=case_files('train');dev=case_files('development');assert len(paths)==33408 and len(dev)==8352
        bundle=Bundle(seed);fit=torch.load(E24/'fit.pt',weights_only=True);B=fit['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long()
        params=[p for p in bundle.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=1e-4);rng=np.random.default_rng(seed)
        schedule=np.concatenate([rng.permutation(len(paths)) for _ in range(2)]);start=0;curve=[];coverage=collections.Counter();checks=[];gradient_coverage=collections.Counter()
        source=dict(protocol=digest(OUT/'protocol.json'),dataset=digest(OUT/'dataset_audit.json'),training_code=digest(__file__),warm=digest(E24/f'training/A_s{seed}/selected.pt'))
        if (root/'latest.pt').exists():
            rec=torch.load(root/'latest.pt',map_location='cuda',weights_only=False);assert rec['source']==source
            bundle.load_state_dict(rec['model']);opt.load_state_dict(rec['optimizer']);start=rec['update'];coverage.update(rec['coverage']);curve=rec['curve'];checks=rec['checks'];rng.bit_generator.state=rec['rng'];gradient_coverage.update(rec['gradient_coverage'])
        for u in range(start,len(schedule)):
            d=load(paths[schedule[u]]);m=bundle.layers[d['key']];s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale'])
            with torch.no_grad():
                mask,r,a=route(d['q'][qs],s,m,d['own']);ff,_=RouteOps.features(d['q'][qs][None],s)
                risk=(ff*m.theta[:,None,None]).sum(-1).mean(1)[0].cpu().numpy()
                mask,r,a=getstate(c,d,mask,r,a,u%10,risk,rng)
            approx=objective(c,d['N'],d['Z'],d['dense'],mask);cp={n:v[qs] for n,v in c.items()};z,uv,beta=bundle.reader(d['q'][qs],s,cp)
            zt,ut,bt,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ'])
            res=nn.functional.smooth_l1_loss(z/fit['zscale'],zt[qs].detach()/fit['zscale'])+nn.functional.smooth_l1_loss(beta/fit['bscale'],bt[qs].detach()/fit['bscale'])
            with torch.no_grad():g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a)
            pg=predicted(z,uv,cp['mass'],mask,r,a,beta)/(d['dense'].square().sum()/64)
            gs=fit['gscale'];target=torch.softmax(torch.cat((g,g.new_zeros(1)))/(.3*gs),0)
            sl=nn.functional.smooth_l1_loss(pg/gs,g/gs)-.1*(target*torch.log_softmax(torch.cat((pg,pg.new_zeros(1)))/gs,0)).sum()
            loss=approx+.1*res+.1*sl;assert torch.isfinite(loss)
            opt.zero_grad();loss.backward()
            for name in ('pool_k','pool_v','pool_feature'):
                grad=getattr(m,name).grad
                if grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.abs().max()>0):gradient_coverage[d['key']+'.'+name]+=1
            gn=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();coverage[d['key']]+=1
            curve.append(dict(update=u,key=d['key'],approx=float(approx),residual=float(res),setloss=float(sl),gradnorm=float(gn)))
            if (u+1)%len(paths)==0:
                rows=evaluate(bundle,dev);epoch=(u+1)//len(paths)-1;atomic(root/f'dev_epoch{epoch}.json',rows)
                ratio=sum(r['initial'] for r in rows)/sum(r['old'] for r in rows);checks.append(dict(epoch=epoch,initial_ratio=ratio))
                atomic(root/f'epoch{epoch}.pt',dict(model=bundle.state_dict(),coverage=dict(coverage),gradient_coverage=dict(gradient_coverage),checks=checks,source=source),True)
            if (u+1)%96==0 or (u+1)%len(paths)==0 or (limit and u+1==limit):
                bundle.check_frozen()
                atomic(root/'latest.pt',dict(model=bundle.state_dict(),optimizer=opt.state_dict(),update=u+1,coverage=dict(coverage),source=source,curve=curve,checks=checks,rng=rng.bit_generator.state,gradient_coverage=dict(gradient_coverage)),True)
                atomic(root/'status.json',dict(status='training',update=u+1,total=len(schedule),positions=len(coverage),seconds=time.perf_counter()-begin));print(seed,u+1,len(coverage),round(time.perf_counter()-begin),flush=True)
            if limit and u+1>=limit:return
        assert len(coverage)==348 and set(coverage.values())=={192}
        assert len(gradient_coverage)==348*3 and min(gradient_coverage.values())>0
        bundle.check_frozen();atomic(root/'complete.json',dict(seed=seed,updates=len(schedule),coverage=dict(coverage),gradient_coverage=dict(gradient_coverage),checks=checks,frozen_exact=True,test_accessed=False))
        atomic(root/'status.json',dict(status='complete',updates=len(schedule),positions=348,seconds=time.perf_counter()-begin))
    finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[17,29],required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();train(a.seed,a.limit)
