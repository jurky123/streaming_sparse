import collections,json,sys,time
from pathlib import Path
import numpy as np
import torch
from policy import initial,pairs,state
from scan import counts
from local_experiments.full_coverage import training_execution as ex, stable_targets as stable, train_full_pset as ps
from local_experiments.full_coverage.common import OUT
from local_experiments.review_v1.ops import RouteOps
from local_experiments.learned_cache.io import atomic
from local_experiments.e30_beta_diagnostics.run import local_scores

root=Path(__file__).parent
idx=int(sys.argv[1]); dest=root/f'budget_{idx}'; dest.mkdir(exist_ok=True)
seed=17
torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=False

table=np.load(root/'policies.npz')[f'{idx}_window_step_layer_head']
original_to_device=ex.to_device
def conditioned(cpu):
    global cell
    d=original_to_device(cpu)
    ev=d['ev']
    cell=int(table[ev['chunk']-2,ev['step'],ev['layer']-1,ev['head']])
    return d
ex.to_device=conditioned

@torch.no_grad()
def route(q,s,m,own):
    f,_=RouteOps.features(q[None],s)
    risk=(f*m.theta[:,None,None]).sum(-1).mean(1)[0]
    own=torch.as_tensor(own,device=q.device)
    mask=initial(risk,own,counts(risk.numel())[cell])
    r,a=pairs(risk,mask,own)
    return mask,r,a,risk

ex.route_with_risk=ps.route_with_risk=route
stable.state=ps.state=state

def dataset(split,n):
    paths=[]
    folders=sorted((root/'records'/split).iterdir())
    assert len(folders)==n
    for folder in folders:
        assert json.loads((folder/'status.json').read_text())['records']==1392
        events=sorted(folder.glob('*_teacher.pt'))
        assert len(events)==1392
        paths.extend(events)
    return paths

def main():
    paths=dataset('train',6); dev=dataset('development',2)
    bundle,phi,reader,fit,_=ps.models(seed)
    bundle.train()
    for m in bundle.layers.values():
        for k in ('pool_k','pool_v','pool_feature'):getattr(m,k).requires_grad_(True)
    bundle.reader.requires_grad_(True)
    reader.load_state_dict(torch.load(OUT/'training/Full_Pset_s17/selected.pt',weights_only=False)['model'])
    B=fit['B'].cuda(); qs=torch.linspace(0,63,16,device='cuda').round().long()
    schedule=np.concatenate([x.permutation(len(paths)) for x in (np.random.default_rng(17),np.random.default_rng(18))])
    original=torch.load(OUT/'training/C_s17/deployment.pt',weights_only=False,map_location='cpu')['model']
    for l in range(1,30):
        for h in range(12):
            for k in ('pool_k','pool_v','pool_feature'):
                assert torch.equal(getattr(bundle.layers[f'l{l}h{h}'],k).cpu()[0],original[f'layers.{l}.{k}'][h])

    def dev_run(label):
        start=time.monotonic(); rows=[]
        with torch.no_grad():
            for u,path in enumerate(dev):
                d=ex.to_device(ex.load_cpu(path))
                s,F,lm,center,mask,r,a,g,norm,c,m=ps.prepare(bundle,phi,d,qs,B.double(),u,seed,False)
                rp,beta=reader(s,F);pred,_=local_scores(rp,beta,lm,center,mask,r,a,norm)
                j=pred.argmax();before=stable.safe_objective(c,d,s,m,mask)
                after=before-(g[j] if pred[j]>0 else 0.)
                assert torch.isfinite(after)
                rows.append([str(path.relative_to(root)),float(before),float(after)])
        atomic(dest/f'dev_{label}.json',dict(seconds=time.monotonic()-start,mean=np.mean([r[1:] for r in rows],axis=0).tolist(),rows=rows))

    if len(sys.argv)>2:
        variant=sys.argv[2]
        if variant=='C_only':
            bundle.load_state_dict(torch.load(dest/'C.pt',weights_only=False)['model'])
        elif variant=='reader_only':
            reader.load_state_dict(torch.load(dest/'reader.pt',weights_only=False)['model'])
        else:
            assert variant=='original'
        dev_run(variant)
        return
    for stage in ('C','Full_Pset'):
        model=bundle if stage=='C' else reader
        if stage=='Full_Pset':bundle.eval().requires_grad_(False)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        checkpoint=dest/(stage+'.pt')
        start=0; elapsed=0.; coverage=collections.Counter(); gradients=collections.Counter(); timings=collections.Counter()
        if checkpoint.exists():
            ck=torch.load(checkpoint,map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
            start=ck['update'];elapsed=ck['seconds'];coverage.update(ck['coverage']);gradients.update(ck['gradients']);timings.update(ck['profile_seconds'])
        begin=time.monotonic()
        for u in range(start,len(schedule)):
            d=ex.to_device(ex.load_cpu(paths[schedule[u]]))
            if stage=='C':
                row,gr,dt=ex.step(bundle,optimizer,d,qs,B,fit,u,np.random.default_rng(seed+u),stable=True,profile=u%128==0)
                gradients.update(gr);timings.update(dt)
            else:
                s,F,lm,center,mask,r,a,g,norm,_,_=ps.prepare(bundle,phi,d,qs,B.double(),u,seed)
                rp,beta=reader(s,F);pg,_=local_scores(rp,beta,lm,center,mask,r,a,norm)
                loss=ps.set_loss(pg,g,ps.rms_floor(g,(0,),fit['gscale']).reshape(()))
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(reader.parameters(),1,error_if_nonfinite=True);optimizer.step()
            coverage[d['key']]+=1
            if (u+1)%256==0 or u+1==len(schedule):
                torch.cuda.synchronize();meta=dict(update=u+1,total=len(schedule),seconds=elapsed+time.monotonic()-begin,coverage=dict(coverage),gradients=dict(gradients),profile_seconds=dict(timings))
                atomic(checkpoint,dict(**meta,model=model.state_dict(),optimizer=optimizer.state_dict()),True)
                atomic(dest/(stage+'.json'),meta)
        assert len(coverage)==348 and set(coverage.values())=={48}
        if stage=='C':
            assert len(gradients)==1044
            ex.check_frozen_fast(bundle)
            for l in range(1,30):
                for k in ('pool_k','pool_v','pool_feature'):
                    original[f'layers.{l}.{k}']=torch.cat([getattr(bundle.layers[f'l{l}h{h}'],k).detach().cpu() for h in range(12)])
            atomic(dest/'summary.pt',dict(model=original),True)
        else:atomic(dest/'reader.pt',dict(model=reader.state_dict()),True)
    dev_run('trained')

if __name__=='__main__':main()
