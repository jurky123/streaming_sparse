"""Exercise both original optimization stages on the new page partition; no deployment export."""
import json,time,sys
from pathlib import Path
import torch,numpy as np
from scan import ROOT,counts
from local_experiments.full_coverage import training_execution as ex,stable_targets as stable
from local_experiments.full_coverage.train import Bundle
from local_experiments.full_coverage.train_full_pset import set_loss
from local_experiments.full_coverage.common import E24,ROOT as REPO
from local_experiments.e24_joint_representatives.core import summary,contributions
from local_experiments.e28_local_response.run import summary_output
from local_experiments.e30_beta_diagnostics.run import local_scores
from local_experiments.e27_adaptive_query.core import AdaptivePhi
from local_experiments.e29_local_reader.core import LocalReader
from local_experiments.e27_adaptive_query.coefficients import rms_floor
from local_experiments.review_v1.ops import RouteOps
def route(q,s,m,own):
    f,_=RouteOps.features(q[None],s)
    risk=(f*m.theta[:,None,None]).sum(-1).mean(1)[0];p=risk.numel()
    a,b=counts(p)[11]
    hp=p-72
    pinned=torch.arange(p,device=q.device)<24;pinned[own]=True
    mask=pinned.clone()
    mask[hp+risk[hp:].masked_fill(pinned[hp:],-torch.inf).topk(a-1).indices]=True
    mask[24+risk[24:hp].topk(b).indices]=True
    r=risk.masked_fill(~mask | pinned,torch.inf).argsort(stable=True)[:min(16,a+b-1)]
    add=risk.masked_fill(mask,-torch.inf).argsort(descending=True,stable=True)[:32]
    rr,aa=torch.meshgrid(r,add,indexing='ij');valid=(rr<hp)==(aa<hp)
    if len(sys.argv)>1 and not valid.any():
        # Separate training pilot: allocate eight removals and sixteen additions
        # per partition when the deployment pool yields no positive/negative pair.
        ri=[];ai=[]
        for part in (torch.arange(p,device=q.device)<hp,torch.arange(p,device=q.device)>=hp):
            elig=(mask & ~pinned & part).nonzero().flatten()
            other=(~mask & part).nonzero().flatten()
            rm=elig[risk[elig].argsort(stable=True)[:8]]
            ad=other[risk[other].argsort(stable=True,descending=True)[:16]]
            rm,ad=torch.meshgrid(rm,ad,indexing='ij');ri.append(rm.flatten());ai.append(ad.flatten())
        return mask,torch.cat(ri),torch.cat(ai),risk.detach().cpu().numpy()
    return mask,rr[valid],aa[valid],risk.detach().cpu().numpy()
def run():
    folder=ROOT/'train_pilot';torch.set_num_threads(2);torch.manual_seed(17)
    ex.route_with_risk=route
    # Use the Risk state in this short integration test. Other state modes
    # require quota-preserving teacher/random proposals in a production run.
    stable.state=lambda c,d,s,m,mask,r,a,*args:(mask,r,a)
    fit=torch.load(E24/'fit.pt',weights_only=True);B=fit['B'].cuda()
    qs=torch.linspace(0,63,16,device='cuda').round().long();bundle=Bundle(17)
    opt=torch.optim.Adam([p for p in bundle.parameters() if p.requires_grad],lr=1e-4)
    records=[ex.to_device(ex.load_cpu(folder/f'c{c}_teacher.pt')) for c in (2,6,20,40)]
    log=[]
    for u,d in enumerate(records):
        with torch.no_grad():
            m=bundle.layers[d['key']]; sm=summary(m,d['k'],d['v'],d['hp'])
            _,r,_,_=route(d['q'][qs],sm,m,d['own'])
        if r.numel()==0:
            log.append(dict(stage='C',chunk=d['ev']['chunk'],skipped='no legal same-part swaps in production candidate pool'))
            continue
        row,gr,times=ex.step(bundle,opt,d,qs,B,fit,u,np.random.default_rng(u),reuse_risk=True,profile=True,stable=True)
        assert len(gr)==3
        log.append(dict(stage='C',chunk=d['ev']['chunk'],metrics=row,gradient=gr,seconds=times))
    phi=AdaptivePhi().cuda();phi.load_state_dict(torch.load(REPO/'artifacts/e27_adaptive_query_v1/s17/selected.pt',weights_only=False)['model']);phi.requires_grad_(False)
    reader=LocalReader().cuda()
    state=torch.load(REPO/'artifacts/e48_tail_state_v1/s17/fullset_final.pt',map_location='cuda',weights_only=False);reader.load_state_dict(state.get('model',state))
    opt=torch.optim.Adam(reader.parameters(),lr=1e-4)
    for d in records:
        torch.cuda.synchronize();t=time.monotonic()
        with torch.no_grad():
            m=bundle.layers[d['key']];s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale'])
            mask,r,a,risk=route(d['q'][qs],s,m,d['own'])
            if r.numel()==0:
                log.append(dict(stage='Full_Pset',chunk=d['ev']['chunk'],skipped='no legal same-part swaps in production candidate pool'))
                continue
            g=stable.true_gains(c,d,s,m,mask,r,a)
            feats=phi(d['q'][qs],s)
            center=(summary_output(c,s,m.basis)[qs]-c['o'][qs,None])@B.double()
            lm=c['logzh'][qs]-c['A'][qs].log()[:,None]
        rp,beta=reader(s,feats);pg,_=local_scores(rp,beta,lm,center,mask,r,a,d['dense'].square().sum()/64+1e-12)
        loss=set_loss(pg,g,rms_floor(g,(0,),fit['gscale']).reshape(()))
        opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(reader.parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        log.append(dict(stage='Full_Pset',chunk=d['ev']['chunk'],loss=float(loss),grad_norm=float(gn),seconds=time.monotonic()-t))
    (folder/('updates_stratified.json' if len(sys.argv)>1 else 'updates.json')).write_text(json.dumps(log,indent=2))
run()
