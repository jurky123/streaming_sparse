"""Continue the exact original Adam sequence with mmap/prefetch and shared Risk."""
import argparse,collections,json,signal,statistics,time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .train import Bundle
from .training_execution import Prefetch,step,check_frozen_fast
from .async_validation import Validation
from ..learned_cache.io import RunLock


def train(seed,eval_gpus,validated_execution_upgrade=False):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);torch.backends.cuda.matmul.allow_tf32=False
    root=OUT/'training'/f'C_s{seed}';lock=RunLock(root);prefetch=None;stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        contract=OUT/'execution_optimization/prefix_contract.json';proof=json.loads(contract.read_text())
        assert proof['passed'] and proof['prefetch_enabled'] and proof['model_bitwise_equal'] and proof['adam_bitwise_equal'] and proof['rng_equal']
        assert proof['source']['reference_code']==digest(Path(__file__).with_name('train.py')) and proof['source']['execution_code']==digest(Path(__file__).with_name('training_execution.py'))
        assert proof['source']['numerics_code']==digest(Path(__file__).with_name('stable_targets.py'))
        numeric_proof=OUT/'execution_optimization/nonfinite/repair_validation.json'
        repair=json.loads(numeric_proof.read_text());assert repair['passed'] and repair['no_clipping'] and repair['no_skipped_events']
        gradient_proof=OUT/'execution_optimization/gradient_failure/repair_validation.json'
        gp=json.loads(gradient_proof.read_text());assert gp['passed'] and gp['gradient_update_finite'] and gp['no_skipped_event'] and gp['no_target_clipping']
        assert gp['numerics_code']==digest(Path(__file__).with_name('stable_targets.py'))
        implementation=dict(name='prefetch_shared_risk_async_eval_v1',code={p:digest(Path(__file__).with_name(p)) for p in ('train_optimized.py','training_execution.py','async_validation.py','evaluate_parallel.py','stable_targets.py')},contract=digest(contract),numerical_repair=digest(numeric_proof),gradient_repair=digest(gradient_proof))
        paths=case_files('train');dev=case_files('development');assert len(paths)==33408 and len(dev)==8352
        source=dict(protocol=digest(OUT/'protocol.json'),dataset=digest(OUT/'dataset_audit.json'),training_code=digest(Path(__file__).with_name('train.py')),warm=digest(E24/f'training/A_s{seed}/selected.pt'))
        rec=torch.load(root/'latest.pt',map_location='cuda',weights_only=False);assert rec['source']==source
        if 'execution_impl' in rec and rec['execution_impl']['code']!=implementation['code']:
            assert validated_execution_upgrade,'Execution code changed; validate before replacing'
            atomic(root/f'execution_upgrade_{rec["update"]}.json',dict(previous=rec['execution_impl'],current=implementation,update=rec['update']))
        bundle=Bundle(seed);bundle.load_state_dict(rec['model']);fit=torch.load(E24/'fit.pt',weights_only=True);B=fit['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long()
        params=[p for p in bundle.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=1e-4);opt.load_state_dict(rec['optimizer'])
        rng=np.random.default_rng(seed);schedule=np.concatenate([rng.permutation(len(paths)) for _ in range(2)]);rng.bit_generator.state=rec['rng'];start=rec['update']
        coverage=collections.Counter(rec['coverage']);gradient=collections.Counter(rec['gradient_coverage']);curve=rec['curve'];validation=Validation(root,seed,eval_gpus,rec['checks'])
        old_status=json.loads((root/'status.json').read_text());prior=rec.get('elapsed_seconds',old_status.get('seconds',0.));begin=time.perf_counter();durations=collections.deque(maxlen=384);wall_markers=collections.deque([(start,begin)],maxlen=8)
        prefetch=Prefetch(paths,schedule,start=start,workers=2,depth=8)
        print('RESUME',seed,start,'preserving original weights, Adam, RNG and schedule',flush=True)
        def save(update,status='training'):
            check_frozen_fast(bundle);elapsed=prior+time.perf_counter()-begin
            atomic(root/'latest.pt',dict(model=bundle.state_dict(),optimizer=opt.state_dict(),update=update,coverage=dict(coverage),source=source,curve=curve,checks=validation.checks,rng=rng.bit_generator.state,gradient_coverage=dict(gradient),execution_impl=implementation,elapsed_seconds=elapsed),True)
            now=time.perf_counter();wall_markers.append((update,now))
            compute_mean=statistics.mean(durations) if durations else None
            delta_updates=update-wall_markers[0][0]
            mean=(now-wall_markers[0][1])/delta_updates if delta_updates else compute_mean
            atomic(root/'status.json',dict(status=status,update=update,total=len(schedule),positions=len(coverage),seconds=elapsed,session_seconds=time.perf_counter()-begin,session_start_update=start,seconds_per_step_recent=mean,seconds_per_step_compute=compute_mean,eta_training_seconds=(len(schedule)-update)*mean if mean else None,evaluation_pending=validation.pending(),evaluation_gpus=eval_gpus,implementation=implementation['name'],numeric_repaired_events=sum('numeric_repairs' in row for row in curve),pid=__import__('os').getpid(),updated=time.time()))
            print(seed,update,len(coverage),round(elapsed), 'step_s',round(mean,4) if mean else None,flush=True)
        for u in range(start,len(schedule)):
            tick=time.perf_counter();d=prefetch.get(u);row,grads,_=step(bundle,opt,d,qs,B,fit,u,rng,reuse_risk=True,stable=True)
            curve.append(row);coverage[d['key']]+=1;gradient.update(grads);durations.append(time.perf_counter()-tick)
            if (u+1)%len(paths)==0:
                epoch=(u+1)//len(paths)-1;check_frozen_fast(bundle)
                snapshot=dict(model=bundle.state_dict(),coverage=dict(coverage),gradient_coverage=dict(gradient),source=source,execution_impl=implementation,epoch=epoch,update=u+1)
                atomic(root/f'epoch{epoch}_weights.pt',snapshot,True)
            if (u+1)%96==0 or stop[0] or (u+1)%len(paths)==0:
                validation.poll();save(u+1)
            if stop[0]:print('STOPPED_AT_CHECKPOINT',seed,u+1,flush=True);return
        assert len(coverage)==348 and set(coverage.values())=={192}
        assert len(gradient)==348*3 and min(gradient.values())>0
        prefetch.close();prefetch=None
        while validation.pending():
            validation.poll();save(len(schedule),'validation')
            if stop[0]:return
            time.sleep(20)
        check_frozen_fast(bundle);save(len(schedule),'complete')
        atomic(root/'complete.json',dict(seed=seed,updates=len(schedule),coverage=dict(coverage),gradient_coverage=dict(gradient),checks=validation.checks,frozen_exact=True,test_accessed=False,execution_impl=implementation))
    except BaseException as exc:
        status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        status.update(status='failed',error=repr(exc),attempted_update=locals().get('u'),updated=time.time())
        if 'u' in locals() and 'schedule' in locals():status['event']=str(paths[schedule[u]].relative_to(OUT))
        atomic(root/'status.json',status);atomic(root/'failure.json',status);raise
    finally:
        if prefetch is not None:prefetch.close()
        lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=[17,29]);p.add_argument('--eval-gpus',type=int,nargs='+');p.add_argument('--validated-execution-upgrade',action='store_true');a=p.parse_args();train(a.seed,a.eval_gpus or ([2,3,4] if a.seed==17 else [5,6,7]),a.validated_execution_upgrade)
