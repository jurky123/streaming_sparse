"""Real four-input serial versus multi-GPU gradient/AdamW audit."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from migration.runtime import multigpu,router_fast
from streaming_sparse.v08.common import atomic_json,nmse,paired_seed,sha,source_identity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',init_method=os.environ.get('SPARSE_RENDEZVOUS'),
        rank=rank,world_size=world,timeout=timedelta(minutes=60))
    paired_seed(0)
    from streaming_sparse.v08.execution import accelerate
    from streaming_sparse.v08.evaluate import load_network
    from streaming_sparse.v08.lora import configure
    from streaming_sparse.v08.replay import ReplayDataset,BalancedReplay
    from streaming_sparse.capture.capture_runner import sdpa_backend
    accelerate.install();router_fast.install()
    network,state=load_network(args.checkpoint);configure(network.model,network.router,'F-K')
    params=[p for p in network.router.parameters()if p.requires_grad]
    names=[n for n,p in network.router.named_parameters()if p.requires_grad]
    initial=copy.deepcopy(network.router.state_dict())
    ds=ReplayDataset('train',require_complete=True);sampler=BalancedReplay(ds,0)
    sampler.load_state_dict(state['sampler']);indices=sampler.sample(args.batch_size)
    def calculate(selected):
        totals=dict(output=0.,att_aux=0.,cal_aux=0.,anchor=0.)
        for i in selected:
            sample=ds.load(i)
            result=network.replay(sample,checkpoint=True,collect_aux=True)
            out=nmse(result['x0'],sample['target'].to('cuda'))
            loss=out+.1*result['att_aux']+.01*result['cal_aux']
            (loss/args.batch_size).backward()
            for name,v in [('output',out),('att_aux',result['att_aux']),('cal_aux',result['cal_aux'])]:totals[name]+=float(v.detach())/args.batch_size
            del result,loss,out
        return totals
    report=dict(passed=False,source_sha256=source_identity(),world_size=world,global_batch=args.batch_size,
        runtime_sha256=router_fast.identity(),router_adapter_sha256=sha(Path(router_fast.__file__)),
        samples=[ds.rows[i]['sample_id']for i in indices])
    try:
        with sdpa_backend():
            if rank==0:
                start=time.perf_counter();totals_serial=calculate(indices);torch.cuda.synchronize()
                serial_seconds=time.perf_counter()-start
                gradients=[p.grad.detach().cpu().clone()if p.grad is not None else None for p in params]
                opt=torch.optim.AdamW(params,lr=3e-7,weight_decay=.01)
                norm_serial=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
                opt.step();serial_params=[p.detach().cpu().clone()for p in params]
                serial_optimizer=copy.deepcopy(opt.state_dict());del opt
                print(json.dumps(dict(event='serial_complete',seconds=serial_seconds,losses=totals_serial)),flush=True)
            network.router.load_state_dict(initial)
            for p in params:p.grad=None
            dist.barrier();start=time.perf_counter()
            totals_parallel=calculate(indices[rank::world])
            multigpu.reduce_update(totals_parallel,params,[])
            torch.cuda.synchronize();seconds=torch.tensor(time.perf_counter()-start,device='cuda',dtype=torch.float64)
            dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
            if rank==0:
                maximum=0.;squared_diff=0.;squared_ref=0.
                for name,p,reference in zip(names,params,gradients):
                    assert(p.grad is None)==(reference is None),name
                    if reference is not None:
                        actual=p.grad.detach().cpu()
                        torch.testing.assert_close(actual,reference,atol=5e-5,rtol=2e-4)
                        maximum=max(maximum,float((actual-reference).abs().max()))
                        squared_diff+=float((actual-reference).double().square().sum())
                        squared_ref+=float(reference.double().square().sum())
                for key in totals_serial:
                    assert abs(totals_serial[key]-totals_parallel[key])<=1e-6,(key,totals_serial,totals_parallel)
                opt=torch.optim.AdamW(params,lr=3e-7,weight_decay=.01)
                norm_parallel=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step()
                param_max=0.
                for p,ref in zip(params,serial_params):
                    torch.testing.assert_close(p.detach().cpu(),ref,atol=1e-6,rtol=2e-4)
                    param_max=max(param_max,float((p.detach().cpu()-ref).abs().max()))
                for index,entry in serial_optimizer['state'].items():
                    for key,value in entry.items():
                        torch.testing.assert_close(value.cpu(),opt.state_dict()['state'][index][key].cpu(),atol=5e-5,rtol=2e-4)
                report.update(passed=True,serial_seconds=serial_seconds,parallel_seconds=float(seconds),
                    speedup=serial_seconds/float(seconds),gradient_max_abs=maximum,
                    gradient_relative_l2=(squared_diff/max(squared_ref,1e-30))**.5,
                    adamw_parameter_max_abs=param_max,optimizer_moments_checked=True,
                    norm_serial=norm_serial,norm_parallel=norm_parallel,
                    serial_losses=totals_serial,parallel_losses=totals_parallel,
                    sampler_fingerprint=sampler.state_dict()['fingerprint'])
                atomic_json(args.output,report)
                print(json.dumps(report),flush=True)
        dist.barrier()
    except BaseException as error:
        if rank==0:report['error']=repr(error);atomic_json(args.output,report)
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
