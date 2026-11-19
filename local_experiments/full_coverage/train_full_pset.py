"""Full-coverage frozen-C Full_Pset adaptation, two-rank synchronous batch=2."""
import argparse,collections,json,os,signal,time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import ROOT,OUT,E24,atomic,digest,case_files
from .train import Bundle
from .training_execution import Prefetch,route_with_risk
from .stable_targets import state,true_gains,safe_objective
from ..e24_joint_representatives.core import summary,contributions
from ..e29_local_reader.core import LocalReader
from ..e27_adaptive_query.core import AdaptivePhi
from ..e27_adaptive_query.coefficients import rms_floor
from ..e28_local_response.run import summary_output
from ..e30_beta_diagnostics.run import local_scores
from ..learned_cache.io import RunLock


def models(seed):
    root=OUT/'training'/f'C_s{seed}';ex=json.loads((root/'export.json').read_text());assert ex['passed'] and ex['positions']==348
    bundle=Bundle(seed);ck=root/f'epoch{ex["epoch"]}.pt';bundle.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model']);bundle.eval().requires_grad_(False)
    phi=AdaptivePhi().cuda();pp=ROOT/f'artifacts/e27_adaptive_query_v1/s{seed}/selected.pt';phi.load_state_dict(torch.load(pp,map_location='cuda',weights_only=False)['model']);phi.eval().requires_grad_(False)
    reader=LocalReader().cuda();rp=ROOT/f'artifacts/e48_tail_state_v1/s{seed}/fullset_final.pt';rec=torch.load(rp,map_location='cuda',weights_only=False);reader.load_state_dict(rec.get('model',rec))
    fit=torch.load(E24/'fit.pt',weights_only=True)
    source=dict(C=digest(ck),deployment=digest(root/'deployment.pt'),phi=digest(pp),reader_warm=digest(rp),B16=digest(E24/'fit.pt'),dataset=digest(OUT/'dataset_audit.json'),code=digest(__file__))
    return bundle,phi,reader,fit,source


@torch.no_grad()
def prepare(bundle,phi,d,qs,B,index,seed,training=True):
    m=bundle.layers[d['key']];s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale'])
    mask,r,a,risk=route_with_risk(d['q'][qs],s,m,d['own'])
    if training:mask,r,a=state(c,d,s,m,mask,r,a,index%10,risk,np.random.default_rng(np.random.SeedSequence([seed,index])))
    g=true_gains(c,d,s,m,mask,r,a);ap=summary_output(c,s,m.basis)
    center=(ap[qs]-c['o'][qs,None])@B
    lm=c['logzh'][qs]-c['A'][qs].log()[:,None]
    return s,phi(d['q'][qs],s),lm,center,mask,r,a,g,d['dense'].square().sum()/64+1e-12,c,m


def set_loss(pg,g,gs):
    target=torch.softmax(torch.cat([g,g.new_zeros(1)])/(.3*gs),0)
    rank=-(target*torch.log_softmax(torch.cat([pg,pg.new_zeros(1)])/gs,0)).sum()
    return torch.nn.functional.smooth_l1_loss(pg/gs,g/gs)+.1*rank


def train(seed,limit=0):
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
    torch.cuda.set_device(rank);torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=15));root=OUT/'training'/f'Full_Pset_s{seed}';root.mkdir(parents=True,exist_ok=True)
    lock=RunLock(root) if rank==0 else None;prefetch=None;stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        bundle,phi,reader,fit,source=models(seed);model=DDP(reader,device_ids=[rank]);opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        paths=case_files('train');assert len(paths)==33408
        rng=np.random.default_rng(seed);schedule=np.concatenate([rng.permutation(len(paths)) for _ in range(2)]);assert len(schedule)%world==0
        total=len(schedule)//world;start=0;coverage=collections.Counter();curve=[]
        if (root/'latest.pt').exists():
            rec=torch.load(root/'latest.pt',map_location='cuda',weights_only=False);assert rec['source']==source
            model.module.load_state_dict(rec['model']);opt.load_state_dict(rec['optimizer']);start=rec['update'];coverage.update(rec['coverage']);curve=rec['curve']
        protocol=dict(seed=seed,stage='Full_Pset',positions=348,epochs=2,events=66816,effective_batch=2,optimizer_updates=total,lr=1e-4,betas=[.9,.999],eps=1e-8,gradient_clip=1,loss='original E48 pure Set-only: smoothL1 + .1 soft ranking, Q64 gain targets / Q64 dense energy; M16 readout',state_cycle='6 Risk / 2 teacher / 2 random; event-index RNG',trainable='shared LocalReader (51157 parameters)',frozen=['full-coverage C','Risk','Adaptive phi','B16','backbone'],initialization='E48 Full_Pset final paired seed',selection='predeclared final; development evaluation, no Test access',not_serial_C_continuation=True,source=source)
        if rank==0:
            if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
            atomic(root/'protocol.json',protocol)
        B=fit['B'].cuda().double();qs=torch.linspace(0,63,16,device='cuda').round().long();prefetch=Prefetch(paths,schedule[rank::world],start=start,workers=2,depth=4)
        frozen={k:v.detach().cpu().clone() for k,v in bundle.state_dict().items()};fphi={k:v.detach().cpu().clone() for k,v in phi.state_dict().items()};begin=time.perf_counter()
        def save(update,status='training'):
            # Counts are global and identical on both ranks.
            if rank==0:
                elapsed=time.perf_counter()-begin;rate=elapsed/max(1,update-start)
                atomic(root/'latest.pt',dict(model=model.module.state_dict(),optimizer=opt.state_dict(),update=update,coverage=dict(coverage),curve=curve,source=source),True)
                atomic(root/'status.json',dict(status=status,seed=seed,update=update,total=total,events_seen=update*world,positions=len(coverage),seconds=elapsed,seconds_per_update=rate,eta_training_seconds=(total-update)*rate,pid=os.getpid(),world_size=world,effective_batch=world,updated=time.time()))
                print('FULL_PSET',seed,update,total,'positions',len(coverage),'step_s',round(rate,4),flush=True)
        save(start)
        for u in range(start,total):
            d=prefetch.get(u);s,F,lm,center,mask,r,a,g,norm,_,_=prepare(bundle,phi,d,qs,B,u*world+rank,seed)
            rp,beta=model(s,F);pg,_=local_scores(rp,beta,lm,center,mask,r,a,norm);gs=rms_floor(g,(0,),fit['gscale']).reshape(())
            loss=set_loss(pg,g,gs);assert torch.isfinite(loss),(seed,u,rank,'nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
            metrics=torch.tensor([float(loss),float(gn)],device='cuda',dtype=torch.float64);dist.all_reduce(metrics);metrics/=world
            for j in range(world):
                p=paths[schedule[u*world+j]];parts=p.stem.split('_');coverage[parts[1]+parts[3]]+=1
            if rank==0:curve.append(dict(update=u+1,setloss=float(metrics[0]),gradnorm=float(metrics[1])))
            stopping=torch.tensor(int(stop[0]),device='cuda');dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
            if (u+1)%48==0 or stopping or u+1==total or (limit and u+1>=limit):save(u+1)
            if stopping or (limit and u+1>=limit):return
        assert len(coverage)==348 and set(coverage.values())=={192},dict(coverage)
        assert all(torch.equal(v.cpu(),frozen[k]) for k,v in bundle.state_dict().items())
        assert all(torch.equal(v.cpu(),fphi[k]) for k,v in phi.state_dict().items())
        if rank==0:
            atomic(root/'final.pt',dict(model=model.module.state_dict(),source=source,coverage=dict(coverage),updates=total),True)
            atomic(root/'training_complete.json',dict(seed=seed,updates=total,coverage=dict(coverage),frozen_C_exact=True,frozen_phi_exact=True,test_accessed=False,source=source))
        save(total,'trained_pending_validation')
    except BaseException as exc:
        atomic(root/f'failure_rank{rank}.json',dict(error=repr(exc),rank=rank,update=locals().get('u'),event=str(paths[schedule[u*world+rank]]) if 'u' in locals() else None,updated=time.time()))
        if rank==0:atomic(root/'status.json',dict(status='failed',error=repr(exc),updated=time.time(),pid=os.getpid()))
        raise
    finally:
        if prefetch:prefetch.close()
        if lock:lock.close()
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(seed):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    root=OUT/'training'/f'Full_Pset_s{seed}';assert (root/'training_complete.json').exists()
    bundle,phi,model,fit,source=models(seed);rec=torch.load(root/'final.pt',map_location='cuda',weights_only=False);assert rec['source']==source;model.load_state_dict(rec['model']);model.eval()
    paths=case_files('development');B=fit['B'].cuda().double();qs=torch.linspace(0,63,16,device='cuda').round().long();rows=[];coverage=collections.Counter();prefetch=Prefetch(paths,np.arange(len(paths)),workers=2,depth=4)
    try:
        for i in range(len(paths)):
            d=prefetch.get(i);s,F,lm,center,mask,r,a,g,norm,c,m=prepare(bundle,phi,d,qs,B,i,seed,False);rp,beta=model(s,F);sc,_=local_scores(rp,beta,lm,center,mask,r,a,norm);j=int(sc.argmax());initial=float(safe_objective(c,d,s,m,mask))
            rows.append(dict(event=str(paths[i].relative_to(OUT)),key=d['key'],initial=initial,executed_gain=float(g[j]) if sc[j]>0 else 0.,oracle_gain=max(0.,float(g.max())),execute=bool(sc[j]>0)));coverage[d['key']]+=1
            if (i+1)%96==0:atomic(root/'evaluation_status.json',dict(status='evaluating',events=i+1,total=len(paths),pid=os.getpid(),updated=time.time()))
        assert len(rows)==8352 and len(coverage)==348 and set(coverage.values())=={24}
        atomic(root/'development.json',rows);gain=sum(r['executed_gain'] for r in rows);base=sum(r['initial'] for r in rows)
        report=dict(seed=seed,events=len(rows),positions=348,executed_gain=gain,initial_error=base,relative_improvement=gain/base,test_accessed=False,source=source)
        atomic(root/'complete.json',report);atomic(root/'selected.pt',rec,True);atomic(root/'status.json',dict(status='complete',**report,updated=time.time()))
    finally:prefetch.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=[17,29]);p.add_argument('--limit',type=int,default=0);p.add_argument('--evaluate',action='store_true');a=p.parse_args();evaluate(a.seed) if a.evaluate else train(a.seed,a.limit)
