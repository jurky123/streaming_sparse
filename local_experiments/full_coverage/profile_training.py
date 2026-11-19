"""Discarded single-video prefix replay; exact numerical contract + phase timing."""
import argparse,copy,json,statistics,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,OUT,E24,atomic,digest
from .train import Bundle,load
from .training_execution import load_cpu,to_device,step,Prefetch,check_frozen_fast


def frozen_state(bundle):return {k:v.detach().cpu().clone() for k,v in bundle.state_dict().items()}


def run(seed=17,steps=12):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    source=OUT/f'training/C_s{seed}/latest.pt';ck=torch.load(source,map_location='cpu',weights_only=False)
    paths=[]
    for chunk in (2,5,8):
        for layer in (1,29):
            for head in (0,11):paths.append(sorted((OUT/'data/city_walk_s701').glob(f'c{chunk}_l{layer}_s0_h{head}_t*.pt'))[0])
    paths=paths[:steps];fit=torch.load(E24/'fit.pt',weights_only=True);B=fit['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long();results={};states={};optimizers={}
    for mode in ('legacy','optimized'):
        b=Bundle(seed);b.load_state_dict(ck['model']);opt=torch.optim.Adam([p for p in b.parameters() if p.requires_grad],lr=1e-4);opt.load_state_dict(copy.deepcopy(ck['optimizer']))
        rng=np.random.default_rng(seed);rng.bit_generator.state=copy.deepcopy(ck['rng']);rows=[];times=[];prefetch=Prefetch(paths,np.arange(len(paths)),workers=2,depth=8) if mode=='optimized' else None;start=time.perf_counter()
        for i,path in enumerate(paths):
            torch.cuda.synchronize();a=time.perf_counter();d=load(path) if mode=='legacy' else prefetch.get(i);torch.cuda.synchronize();loading=time.perf_counter()-a
            row,gradient,phases=step(b,opt,d,qs,B,fit,ck['update']+i,rng,reuse_risk=mode=='optimized',profile=True,stable=mode=='optimized')
            rows.append(row);times.append(dict(load=loading,**phases))
        if prefetch is not None:prefetch.close()
        check_start=time.perf_counter()
        if mode=='optimized':check_frozen_fast(b)
        else:b.check_frozen()
        frozen_seconds=time.perf_counter()-check_start
        states[mode]=frozen_state(b);optimizers[mode]=copy.deepcopy(opt.state_dict())
        results[mode]=dict(rows=rows,frozen_check_seconds=frozen_seconds,seconds=time.perf_counter()-start,phase_means={name:statistics.mean(x[name] for x in times) for name in times[0]},steady_phase_means={name:statistics.mean(x[name] for x in times[4:]) for name in times[0]},rng=copy.deepcopy(rng.bit_generator.state))
        del b,opt
    assert results['legacy']['rows']==results['optimized']['rows'],'Step metrics differ'
    assert results['legacy']['rng']==results['optimized']['rng'],'RNG differs'
    assert all(torch.equal(v,states['optimized'][k]) for k,v in states['legacy'].items()),'Model differs'
    for key,state in optimizers['legacy']['state'].items():
        for name,value in state.items():
            other=optimizers['optimized']['state'][key][name]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other,(key,name)
    report=dict(passed=True,prefetch_enabled=True,video='city_walk_s701',steps=len(paths),discarded=True,start_update=ck['update'],model_bitwise_equal=True,adam_bitwise_equal=True,rng_equal=True,metrics_exact=True,results=results,
        source=dict(checkpoint=str(source),reference_code=digest(ROOT/'local_experiments/full_coverage/train.py'),execution_code=digest(ROOT/'local_experiments/full_coverage/training_execution.py'),numerics_code=digest(ROOT/'local_experiments/full_coverage/stable_targets.py')))
    atomic(OUT/'execution_optimization/prefix_contract.json',report)
    print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int,default=12);a=p.parse_args();run(a.seed,a.steps)
