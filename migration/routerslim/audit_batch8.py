"""Baseline versus R1 on the exact next eight sampled inputs and stored AdamW."""
import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from migration.runtime import multigpu,router_fast
from migration.routerslim import r1
from streaming_sparse.v08.common import atomic_json,nmse,paired_seed,sha,source_identity,protocol,tensor_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==8
    torch.cuda.set_device(0);dist.init_process_group('nccl',init_method=os.environ['SPARSE_RENDEZVOUS'],rank=rank,world_size=world,timeout=timedelta(minutes=30))
    paired_seed(0)
    from streaming_sparse.v08.execution import accelerate
    from streaming_sparse.v08.evaluate import load_network
    from streaming_sparse.v08.lora import configure
    from streaming_sparse.v08.replay import ReplayDataset,BalancedReplay
    from streaming_sparse.capture.capture_runner import sdpa_backend
    accelerate.install();router_fast.install();network,state=load_network(args.checkpoint);configure(network.model,network.router,'F-K')
    params=[p for p in network.router.parameters()if p.requires_grad]
    names=[n for n,p in network.router.named_parameters()if p.requires_grad]
    initial=copy.deepcopy(network.router.state_dict());ds=ReplayDataset('train',require_complete=True);sampler=BalancedReplay(ds,0)
    sampler.load_state_dict(state['sampler']);indices=sampler.sample(8);step=state['updates']+1
    report=dict(passed=False,source_sha256=source_identity(),r1_sha256=r1.identity(),checkpoint_sha256=sha(args.checkpoint),
        world_size=8,global_batch=8,update=step,features=dict(variance=True,projection=False,skip_zero_aux=True),
        samples=[ds.rows[i]['sample_id']for i in indices],gradient_tolerance=dict(atol=5e-5,rtol=2e-4))
    if rank==0:
        atomic_json(Path(args.output).parent/'baseline_manifest.json',dict(
            source_sha256=source_identity(),checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha(args.checkpoint),updates=state['updates'],
            backbone_sha256=protocol()['teacher_sha256'],router_tensor_sha256=tensor_digest(state['router'].items()),
            replay_train_sha256=ds.fingerprint(),global_batch=8,pages='continuous Page64 including tail',
            layers=[4,14,24],heads=12,dim=128,prototypes=8,history_tokens=[4680,9360,14040],keep=.2,
            next_update=step,next_inputs=report['samples'],query_policy='All 4680 original independent queries; original 8 selected tiles of 16 for calibration',
            backbone_dtype='bfloat16',router_dtype='float32',budget_gate_vjp_dtype='float64',tf32=False,
            gradient_tolerance=dict(atol=5e-5,rtol=2e-4),output_tolerance=dict(nmse=1e-4,maxabs=.125),
            optimizer='Native AdamW; existing moments, schedule and sampler restored; gradient SUM before clipping'))
    saved=[]
    try:
        for optimized in (False,True):
            if optimized:r1.install(variance=True,projection=False,skip_zero_aux=True);r1.configure_update(network,step)
            else:network._routerslim_skip_att_aux=False
            network.router.load_state_dict(initial)
            for param in params:param.grad=None
            network._eval_history=None;network._eval_sketch.clear()
            totals=dict(output=0.,cal_aux=0.,anchor=0.)
            outputs=[]
            with sdpa_backend():
                for i in indices[rank::world]:
                    sample=ds.load(i);result=network.replay(sample,checkpoint=True,collect_aux=True)
                    out=nmse(result['x0'],sample['target'].cuda());loss=out+0.*result['att_aux']+.01*result['cal_aux']
                    outputs.append(result['x0'].detach().cpu().clone())
                    (loss/8).backward();totals['output']+=float(out)/8;totals['cal_aux']+=float(result['cal_aux'])/8
                    del result,loss,out
                    if step%4==0:
                        dense=network.replay(sample,sparse=False,checkpoint=True,collect_aux=False)
                        anchor=nmse(dense['x0'],sample['target'].cuda())
                        if anchor.requires_grad:(anchor*.4/8).backward()
                        totals['anchor']+=float(anchor)/8;del dense,anchor
                multigpu.reduce_update(totals,params,[])
            gradients=[p.grad.detach().cpu().clone()if p.grad is not None else None for p in params]
            optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01);optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            settings=state['provenance']['settings'];progress=min(1.,max(0.,(step-settings['warmup'])/(2000-settings['warmup'])))
            multiplier=min(1.,step/settings['warmup'])*(.05+.95*(1+math.cos(math.pi*progress))/2)
            for pg in optimizer.param_groups:pg['lr']=pg['initial_lr']*multiplier
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step()
            saved.append(dict(outputs=outputs,gradients=gradients,parameters=[p.detach().cpu().clone()for p in params],optimizer=copy.deepcopy(optimizer.state_dict()),totals=totals,norm=norm))
            del optimizer
            if optimized:r1.uninstall()
            if rank==0:print(json.dumps(dict(event='batch8_pass_complete',optimized=optimized,losses=totals)),flush=True)
        a,b=saved;maxdiff=0.;paramdiff=0.;momentdiff=0.
        for x,y in zip(a['outputs'],b['outputs']):torch.testing.assert_close(x,y,atol=0,rtol=0)
        for n,x,y in zip(names,a['gradients'],b['gradients']):
            assert(x is None)==(y is None),n
            if x is not None:
                torch.testing.assert_close(x,y,atol=5e-5,rtol=2e-4);maxdiff=max(maxdiff,float((x-y).abs().max()))
        for x,y in zip(a['parameters'],b['parameters']):
            torch.testing.assert_close(x,y,atol=1e-6,rtol=2e-4);paramdiff=max(paramdiff,float((x-y).abs().max()))
        for index,entry in a['optimizer']['state'].items():
            for key,x in entry.items():
                y=b['optimizer']['state'][index][key];torch.testing.assert_close(x,y,atol=5e-5,rtol=2e-4)
                momentdiff=max(momentdiff,float((x-y).abs().max()))
        for name,x in a['totals'].items():assert abs(x-b['totals'][name])<=1e-6
        report.update(passed=True,gradient_maxabs=maxdiff,adamw_parameter_maxabs=paramdiff,adamw_moment_maxabs=momentdiff,
            losses=[a['totals'],b['totals']],norms=[a['norm'],b['norm']],sampler_unchanged=True,anchor_computed=step%4==0,all_eight_outputs_bitwise_equal=True)
        if rank==0:atomic_json(args.output,report);print(json.dumps(report),flush=True)
        dist.barrier()
    except BaseException as error:
        if rank==0:report['error']=repr(error);atomic_json(args.output,report)
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
