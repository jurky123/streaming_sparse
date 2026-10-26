"""Full-input serial versus optimized sample-parallel gradient/AdamW audit."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from migration.paired_router import runtime
from migration.runtime import multigpu,router_fast
from migration.routerslim import r1
from streaming_sparse.v08.common import atomic_json,paired_seed,source_identity,sha,nmse


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--stage',required=True);p.add_argument('--group',required=True);a=p.parse_args()
    manifest,spec,_=runtime.configuration(a.directory,a.stage);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
    torch.cuda.set_device(0);dist.init_process_group('nccl',init_method=os.environ['SPARSE_RENDEZVOUS'],rank=rank,world_size=world,timeout=timedelta(hours=2))
    paired_seed(0)
    from streaming_sparse.v08.execution import accelerate
    from streaming_sparse.v08.lora import configure
    from streaming_sparse.v08.replay import ReplayDataset,BalancedReplay
    from streaming_sparse.capture.capture_runner import sdpa_backend
    accelerate.install();router_fast.install()
    parent=Path(spec['directory'])/f'runs/seed0/warmstart_{a.group[-1]}/best.pt'
    net,state=runtime.load_checkpoint(parent,spec['layers']);configure(net.model,net.router,a.group)
    ds=ReplayDataset('train',require_complete=True);schedule=BalancedReplay(ds,0);indices=schedule.sample(8)
    params=list(net.router.parameters());names=[n for n,p in net.router.named_parameters()];initial=copy.deepcopy(net.router.state_dict())
    report=dict(passed=False,source_sha256=source_identity(),runtime_sha256=router_fast.identity(),paired_runtime_sha256=runtime.identity(),r1_sha256=r1.identity(),
        router_adapter_sha256=sha(router_fast.__file__),world_size=4,global_batch=8,stage=a.stage,group=a.group,layers=spec['layers'],
        samples=[ds.rows[i]['sample_id']for i in indices],parent_sha256=sha(parent),first_step_lr=3e-7,warmup_updates=100)
    directory=Path(spec['directory'])/'audits';output=directory/f'{a.group}.json'
    def calculate(selected):
        totals=dict(output=0.,att_aux=0.,cal_aux=0.,anchor=0.);outputs=[]
        for i in selected:
            sample=ds.load(i);result=net.replay(sample,checkpoint=True,collect_aux=True)
            out=nmse(result['x0'],sample['target'].cuda());loss=out+.1*result['att_aux']+.01*result['cal_aux']
            (loss/8).backward();outputs.append(result['x0'].detach().cpu())
            for k,v in [('output',out),('att_aux',result['att_aux']),('cal_aux',result['cal_aux'])]:totals[k]+=float(v)/8
            del result,loss,out
        return totals,outputs
    try:
        with sdpa_backend():
            if rank==0:
                start=time.perf_counter();serial,out0=calculate(indices);torch.cuda.synchronize();serial_time=time.perf_counter()-start
                grads=[p.grad.detach().cpu().clone()if p.grad is not None else None for p in params]
                opt=torch.optim.AdamW(params,lr=3e-7,weight_decay=.01);torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
                weights=[p.detach().cpu().clone()for p in params];moments=copy.deepcopy(opt.state_dict());del opt
                print(json.dumps(dict(event='serial_complete',seconds=serial_time)),flush=True)
            net.router.load_state_dict(initial)
            for p in params:p.grad=None
            net._eval_history=None;net._eval_sketch.clear();r1.install(variance=True,projection=False,skip_zero_aux=True);r1.configure_update(net,1)
            dist.barrier();start=time.perf_counter();parallel,outputs=calculate(indices[rank::world]);multigpu.reduce_update(parallel,params,[])
            seconds=torch.tensor(time.perf_counter()-start,device='cuda',dtype=torch.float64);dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
            gather=[None]*world if rank==0 else None;dist.gather_object(outputs,gather,dst=0)
            if rank==0:
                for k in serial:assert abs(serial[k]-parallel[k])<1e-6,(k,serial[k],parallel[k])
                for j,out in enumerate(out0):torch.testing.assert_close(out,gather[j%world][j//world],atol=0,rtol=0)
                maximum=0.
                for n,p,g in zip(names,params,grads):
                    assert(p.grad is None)==(g is None),n
                    if g is not None:
                        torch.testing.assert_close(p.grad.cpu(),g,atol=5e-5,rtol=2e-4);maximum=max(maximum,float((p.grad.cpu()-g).abs().max()))
                opt=torch.optim.AdamW(params,lr=3e-7,weight_decay=.01);torch.nn.utils.clip_grad_norm_(params,1.);opt.step();parameter_max=0.
                for p,w in zip(params,weights):
                    torch.testing.assert_close(p.cpu(),w,atol=1e-6,rtol=2e-4);parameter_max=max(parameter_max,float((p.cpu()-w).abs().max()))
                for index,entry in moments['state'].items():
                    for key,v in entry.items():torch.testing.assert_close(v.cpu(),opt.state_dict()['state'][index][key].cpu(),atol=5e-5,rtol=2e-4)
                report.update(passed=True,outputs_bitwise_equal=True,gradient_maxabs=maximum,adamw_parameter_maxabs=parameter_max,
                    native_moments_checked=True,serial_seconds=serial_time,parallel_seconds=float(seconds),losses=parallel)
                atomic_json(output,report);print(json.dumps(report),flush=True)
            dist.barrier()
    except BaseException as error:
        if rank==0:report['error']=repr(error);atomic_json(output,report)
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
