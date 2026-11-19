"""Execution-only helpers: identical event order, losses, FP precision and Adam steps."""
import collections,concurrent.futures,time
import numpy as np
import torch
from torch import nn
from .common import NAMES
from .train import load as legacy_load
from ..review_v1.ops import Summary,RouteOps,Config
from ..e12_swap_prediction.core import pool
from ..e24_joint_representatives.core import summary,contributions,route,gains,objective,labels,predicted
from ..e24_joint_representatives.train import getstate


def load_cpu(path,pin=False):
    # mmap avoids materializing the unused 27 MB per-page output teacher.
    ev=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if ev['split'] not in ('train','development'):raise ValueError('Test sealed')
    kv=torch.load(path.parent/ev['kv_file'],map_location='cpu',weights_only=True,mmap=True)
    if pin:
        ev=dict(ev);kv=dict(kv)
        for name in ('q','N','Z','dense','log_scale','page_logz'):ev[name]=ev[name].pin_memory()
        for name in ('k','v'):kv[name]=kv[name].pin_memory()
    return path,ev,kv


def to_device(cpu,include_old=False):
    path,ev,kv=cpu
    d={n:ev[n].cuda(non_blocking=True).float() if n=='q' else ev[n].cuda(non_blocking=True).double() for n in ('q','N','Z','dense','log_scale')}
    d.update(ev=ev,k=kv['k'][None].cuda(non_blocking=True),v=kv['v'][None].cuda(non_blocking=True),hp=kv['history_pages'],own=ev['own'],key=f'l{ev["layer"]}h{ev["head"]}')
    if include_old:d['old']=Summary(*(kv['summary'][n].cuda() for n in NAMES))
    d['logZ']=ev['page_logz'].cuda(non_blocking=True).double()-d['log_scale'][:,None]
    return d


class Prefetch:
    """Bounded CPU-only mmap prefetch; delivery always follows the fixed schedule."""
    def __init__(self,paths,indices,start=0,workers=2,depth=8):
        self.paths=paths;self.indices=indices;self.next=start;self.submit=start;self.depth=depth
        self.pool=concurrent.futures.ThreadPoolExecutor(workers);self.pending={};self.fill()
    def fill(self):
        while self.submit<len(self.indices) and len(self.pending)<self.depth:
            i=self.submit;self.pending[i]=self.pool.submit(load_cpu,self.paths[self.indices[i]],True);self.submit+=1
    def get(self,i):
        if i!=self.next:raise ValueError('Prefetch cannot reorder optimizer steps')
        value=self.pending.pop(i).result();self.next+=1;self.fill();return to_device(value)
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)


@torch.no_grad()
def route_with_risk(q,s,m,own):
    f,_=RouteOps.features(q[None],s)
    risk=(f*m.theta[:,None,None]).sum(-1).mean(1)[0]
    mask=RouteOps.select(risk[None,None],torch.tensor([[own]],device=q.device),Config())['mask'][0,0]
    cpu=risk.cpu().numpy();r,a=pool({'risk':cpu},mask.cpu().numpy(),own)
    return mask,torch.as_tensor(r,device=q.device),torch.as_tensor(a,device=q.device),cpu


def step(bundle,opt,d,qs,B,fit,u,rng,*,reuse_risk=True,profile=False,stable=False):
    times={};last=[time.perf_counter()]
    def mark(name):
        if profile:
            torch.cuda.synchronize();now=time.perf_counter();times[name]=now-last[0];last[0]=now
    if profile:torch.cuda.synchronize();last[0]=time.perf_counter()
    m=bundle.layers[d['key']];s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale']);mark('summary_contributions')
    with torch.no_grad():
        if reuse_risk:mask,r,a,risk=route_with_risk(d['q'][qs],s,m,d['own'])
        else:
            mask,r,a=route(d['q'][qs],s,m,d['own']);ff,_=RouteOps.features(d['q'][qs][None],s)
            risk=(ff*m.theta[:,None,None]).sum(-1).mean(1)[0].cpu().numpy()
        if stable:
            from .stable_targets import state
            mask,r,a=state(c,d,s,m,mask,r,a,u%10,risk,rng)
        else:mask,r,a=getstate(c,d,mask,r,a,u%10,risk,rng)
    mark('risk_and_state')
    if stable:
        from .stable_targets import safe_objective,true_gains,proxy_gains
        approx=safe_objective(c,d,s,m,mask)
    else:approx=objective(c,d['N'],d['Z'],d['dense'],mask)
    cp={n:v[qs] for n,v in c.items()};z,uv,beta=bundle.reader(d['q'][qs],s,cp)
    zt,ut,bt,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ'])
    res=nn.functional.smooth_l1_loss(z/fit['zscale'],zt[qs].detach()/fit['zscale'])+nn.functional.smooth_l1_loss(beta/fit['bscale'],bt[qs].detach()/fit['bscale'])
    with torch.no_grad():g=true_gains(c,d,s,m,mask,r,a) if stable else gains(c,d['N'],d['Z'],d['dense'],mask,r,a)
    if stable:
        raw,used=proxy_gains(bundle.reader,d['q'][qs],s,cp,mask,r,a,z,uv,beta)
        if used:d.setdefault('numeric_repairs',[]).append('proxy_log_mixture')
    else:raw=predicted(z,uv,cp['mass'],mask,r,a,beta)
    pg=raw/(d['dense'].square().sum()/64)
    gs=fit['gscale'];target=torch.softmax(torch.cat((g,g.new_zeros(1)))/(.3*gs),0)
    sl=nn.functional.smooth_l1_loss(pg/gs,g/gs)-.1*(target*torch.log_softmax(torch.cat((pg,pg.new_zeros(1)))/gs,0)).sum()
    loss=approx+.1*res+.1*sl;assert torch.isfinite(loss);mark('reader_labels_gains_loss')
    params=[p for p in bundle.parameters() if p.requires_grad]
    opt.zero_grad();loss.backward();mark('backward')
    gradient=[]
    for name in ('pool_k','pool_v','pool_feature'):
        grad=getattr(m,name).grad
        if grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.abs().max()>0):gradient.append(d['key']+'.'+name)
    gn=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();mark('gradient_checks_adam')
    row=dict(update=u,key=d['key'],approx=float(approx),residual=float(res),setloss=float(sl),gradnorm=float(gn));mark('logging')
    if d.get('numeric_repairs'):row['numeric_repairs']=d['numeric_repairs']
    return row,gradient,times


def check_frozen_fast(bundle):
    # Build the 2784-entry state dict once, rather than once per frozen tensor.
    state=bundle.layers.state_dict()
    assert all(torch.equal(state[k].cpu(),v) for k,v in bundle.frozen.items())
