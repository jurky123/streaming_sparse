"""Compare execution adapters against the already audited execution path."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from migration.runtime import router_fast
from streaming_sparse.v08.common import HERE, atomic_json, nmse, paired_seed, source_identity, sha
from streaming_sparse.v08.execution import accelerate


def gates():
    from streaming_sparse.v05.primitives import BudgetGate
    report = []
    for history in (4680, 9360, 14040):
        costs = (history-torch.arange((history+63)//64, device='cuda')*64).clamp(0,64).float()
        scores = torch.randn(4680, len(costs), device='cuda')
        gradient = torch.randn_like(scores)
        saved = {}; times = {}
        for enabled in (False, True):
            if enabled: router_fast.install()
            x = scores.clone().requires_grad_(); c = costs.clone().requires_grad_()
            p = BudgetGate.apply(x,c,int(history*.2),.2)
            grads = torch.autograd.grad((p*gradient).sum(), (x,c))
            saved[enabled] = (p.detach(), *[g.detach() for g in grads])
            torch.cuda.synchronize(); start = time.perf_counter()
            for _ in range(5): BudgetGate.apply(scores, costs, int(history*.2), .2)
            torch.cuda.synchronize(); times[str(enabled)] = (time.perf_counter()-start)/5
            router_fast.uninstall()
        for a,b in zip(saved[False],saved[True]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        report.append(dict(history=history, probability_and_both_gradients_bitwise_equal=True,
                           seconds=times,speedup=times['False']/times['True']))
    return report


def full(checkpoint):
    from streaming_sparse.v08.evaluate import load_network
    from streaming_sparse.v08.lora import configure
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.capture.capture_runner import sdpa_backend
    network,state=load_network(checkpoint)
    configure(network.model,network.router,'F-K')
    ds=ReplayDataset('train',require_complete=True)
    params=[(n,p)for n,p in network.named_parameters()if p.requires_grad]
    results=[]
    for history in (4680,9360,14040):
        index=next(i for i,r in enumerate(ds.rows)if r['history_tokens']==history and r['step']==0)
        sample=ds.load(index); saved={}
        for enabled in (False,True):
            if enabled:router_fast.install()
            for _,p in params:p.grad=None
            network._eval_history=None;network._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with sdpa_backend():
                result=network.replay(sample,collect_aux=True,checkpoint=True)
                loss=nmse(result['x0'],sample['target'].to('cuda'))+.1*result['att_aux']+.01*result['cal_aux']
                torch.cuda.synchronize(); forward=time.perf_counter()-start
                loss.backward();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            saved[enabled]=dict(output=result['x0'].detach().cpu(),loss=float(loss),
                gradients={n:p.grad.detach().cpu().clone()for n,p in params if p.grad is not None},
                forward=forward,forward_backward=elapsed,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            print(json.dumps(dict(history=history,fast=enabled,seconds=elapsed,loss=float(loss))),flush=True)
            del result,loss
            router_fast.uninstall()
        a,b=saved[False],saved[True]
        torch.testing.assert_close(a['output'],b['output'],rtol=0,atol=0)
        assert a['gradients'].keys()==b['gradients'].keys()
        difference=0.
        for n,g in a['gradients'].items():
            # The experiment's pre-existing execution audit tolerances, fixed
            # before this adapter. Larger independent query batches may change
            # FP32 accumulation; discrete native outputs remain exact.
            torch.testing.assert_close(g,b['gradients'][n],rtol=2e-4,atol=5e-5)
            difference=max(difference,float((g-b['gradients'][n]).abs().max()))
        for x in saved.values():del x['output'],x['gradients']
        results.append(dict(history=history,sample_id=sample['sample_id'],output_bitwise_equal=True,
                            gradients_within_original_execution_tolerances=True,
                            maximum_gradient_difference=difference,measurements=saved,
                            speedup=a['forward_backward']/b['forward_backward']))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();paired_seed(0);accelerate.install()
    report=dict(passed=False,source_sha256=source_identity(),runtime_sha256=router_fast.identity(),
                router_adapter_sha256=sha(Path(router_fast.__file__)),
                gradient_tolerance=dict(atol=5e-5,rtol=2e-4),score_batch=512,gate_vjp_batch=256)
    try:
        report['gate']=gates();atomic_json(args.output,report)
        report['full']=full(args.checkpoint);report['passed']=True
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        router_fast.uninstall();atomic_json(args.output,report)


if __name__=='__main__':main()
