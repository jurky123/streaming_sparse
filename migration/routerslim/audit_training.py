"""Full original replay, output/gradient/AdamW equivalence at a frozen state."""
import argparse
import copy
import gc
import json
import time
import torch
from migration.routerslim import r1
from migration.runtime import router_fast
from streaming_sparse.v08.common import atomic_json,nmse,paired_seed,source_identity,sha
from streaming_sparse.v08.execution import accelerate


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--history',type=int,required=True);p.add_argument('--steps',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--variant',choices=['all','training','variance','projection','skip'],default='all')
    args=p.parse_args();paired_seed(0);accelerate.install();router_fast.install()
    from streaming_sparse.v08.evaluate import load_network
    from streaming_sparse.v08.lora import configure
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.capture.capture_runner import sdpa_backend
    network,state=load_network(args.checkpoint);configure(network.model,network.router,'F-K')
    ds=ReplayDataset('train',require_complete=True)
    params=[(n,p)for n,p in network.named_parameters()if p.requires_grad]
    report=dict(passed=False,source_sha256=source_identity(),r1_sha256=r1.identity(),checkpoint_sha256=sha(args.checkpoint),
        variant=args.variant,updates=state['updates'],gradient_tolerance=dict(atol=5e-5,rtol=2e-4),cases=[])
    try:
        for step in args.steps:
            idx=next(i for i,row in enumerate(ds.rows)if row['history_tokens']==args.history and row['step']==step)
            sample=ds.load(idx);saved=[]
            for opt in (False,True):
                if opt:
                    r1.install(variance=args.variant in ('all','training','variance'),projection=args.variant in ('all','projection'),skip_zero_aux=args.variant in ('all','training','skip'))
                    r1.configure_update(network,state['updates']+1)
                else:network._routerslim_skip_att_aux=False
                for _,param in params:param.grad=None
                network._eval_history=None;network._eval_sketch.clear();gc.collect()
                torch.cuda.synchronize();start=time.perf_counter()
                with sdpa_backend():
                    result=network.replay(sample,checkpoint=True,collect_aux=True)
                    loss=nmse(result['x0'],sample['target'].cuda())+0.*result['att_aux']+.01*result['cal_aux']
                    loss.backward()
                torch.cuda.synchronize()
                item=dict(output=result['x0'].detach().cpu(),loss=float(loss),cal_aux=float(result['cal_aux']),
                    gradients={n:p.grad.detach().cpu().clone()for n,p in params if p.grad is not None},seconds=time.perf_counter()-start)
                saved.append(item)
                print(json.dumps(dict(event='measurement',history=args.history,step=step,optimized=opt,seconds=item['seconds'],loss=item['loss'])),flush=True)
                del result,loss
                if opt:r1.uninstall()
            a,b=saved;difference=float((a['output'].float()-b['output'].float()).abs().max())
            # Existing execution tolerances; always record whether exact.
            error=float((a['output'].float()-b['output'].float()).square().sum()/a['output'].float().square().sum().clamp_min(1e-30))
            assert difference<=.125 and error<=1e-4,(difference,error)
            assert a['gradients'].keys()==b['gradients'].keys()
            for n,g in a['gradients'].items():torch.testing.assert_close(g,b['gradients'][n],atol=5e-5,rtol=2e-4,msg=lambda m:n+' '+m)
            # Compare one real native AdamW update from the SAME stored moments.
            # F-K optimizer has exactly one router group, in router order.
            updates=[]
            for saved_grad in (a['gradients'],b['gradients']):
                values=[torch.nn.Parameter(p.detach().cpu().clone())for n,p in params]
                optimizer=torch.optim.AdamW(values,lr=3e-5)
                optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
                for (n,_),value in zip(params,values):value.grad=saved_grad.get(n)
                torch.nn.utils.clip_grad_norm_(values,1.,error_if_nonfinite=True)
                optimizer.step()
                updates.append([v.detach()for v in values])
            optimizer_maxabs=max(float((x-y).abs().max())for x,y in zip(*updates))
            for x,y in zip(*updates):torch.testing.assert_close(x,y,atol=5e-5,rtol=2e-4)
            item=dict(history=args.history,step=step,sample_id=sample['sample_id'],output_maxabs=difference,output_nmse=error,
                output_bitwise_equal=torch.equal(a['output'],b['output']),gradient_maxabs=max(float((g-b['gradients'][n]).abs().max())for n,g in a['gradients'].items()),
                adamw_maxabs=optimizer_maxabs,baseline_seconds=a['seconds'],optimized_seconds=b['seconds'],losses=[a['loss'],b['loss']],
                calibration=[a['cal_aux'],b['cal_aux']],parameter_gradient_presence_equal=True)
            report['cases'].append(item);atomic_json(args.output,report);print(json.dumps(item),flush=True)
        report['passed']=True
    except BaseException as error:report['error']=repr(error);raise
    finally:r1.uninstall();atomic_json(args.output,report)


if __name__=='__main__':main()
