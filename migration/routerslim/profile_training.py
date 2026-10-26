"""Frozen-checkpoint full-input training timings and component CUDA profiles."""
import argparse
import csv
import gc
import json
from pathlib import Path
import time
import torch
from streaming_sparse.v08.common import atomic_json,paired_seed,source_identity,sha,nmse


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--no-trace',action='store_true')
    parser.add_argument('--variant',choices=['baseline','all','training','variance','projection','skip'],default='baseline')
    parser.add_argument('--histories',type=int,nargs='+',default=[4680,14040])
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    paired_seed(0)
    from streaming_sparse.v08.evaluate import load_network
    from streaming_sparse.v08.lora import configure
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.execution import accelerate
    from migration.runtime import router_fast
    from migration.routerslim import profile_ranges
    from streaming_sparse.capture.capture_runner import sdpa_backend
    accelerate.install();router_fast.install()
    network,state=load_network(args.checkpoint);configure(network.model,network.router,state['group'])
    if args.variant!='baseline':
        from migration.routerslim import r1
        r1.install(variance=args.variant in ('all','training','variance'),projection=args.variant in ('all','projection'),skip_zero_aux=args.variant in ('all','training','skip'))
        r1.configure_update(network,state['updates']+1)
    ds=ReplayDataset('train',require_complete=True)
    parameters=[p for p in network.parameters()if p.requires_grad]
    report=dict(source_sha256=source_identity(),checkpoint_sha256=sha(args.checkpoint),updates=state['updates'],
        group=state['group'],train_fingerprint=ds.fingerprint(),att_weight=0.,cal_weight=.01,
        backbone_dtype='bfloat16',router_dtype='float32',gate_vjp_dtype='float64',
        previous_execution_sha256=router_fast.identity(),variant=args.variant,samples=[],passed=False,
        timing_policy='Synchronized per complete input; warm CUDA graphs; history rebuilt for every training input; no optimizer mutation; latency trials exclude profiler.')
    def execute(sample,annotated=False):
        for p in parameters:p.grad=None
        network._eval_history=None;network._eval_sketch.clear()
        with torch.profiler.record_function('train.forward'):
            result=network.replay(sample,checkpoint=True,collect_aux=True)
            loss=nmse(result['x0'],sample['target'].to('cuda'))+.0*result['att_aux']+.01*result['cal_aux']
        with torch.profiler.record_function('train.backward'):loss.backward()
        torch.cuda.synchronize()
        return float(loss.detach())
    import numpy as np
    with sdpa_backend():
        for history in args.histories:
            index=next(i for i,r in enumerate(ds.rows)if r['history_tokens']==history and r['step']==0)
            sample=ds.load(index);torch.cuda.synchronize();start=time.perf_counter();cold_loss=execute(sample);cold=time.perf_counter()-start
            timings=[]
            for repeat in range(args.repeats):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();value=execute(sample)
                timings.append(time.perf_counter()-start)
                print(json.dumps(dict(event='timing',history=history,repeat=repeat,seconds=timings[-1],loss=value)),flush=True)
            item=dict(sample_id=sample['sample_id'],history_tokens=history,queries=4680,all_pairs=36,
                cold_seconds=cold,steady_seconds=timings,median_seconds=float(np.median(timings)),
                p90_seconds=float(np.quantile(timings,.9)),loss=value,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            if not args.no_trace:
                profile_ranges.install(network)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False)as prof:
                    execute(sample,True)
                profile_ranges.uninstall()
                breakdown=profile_ranges.summarize(prof);item['profile']=breakdown
                prof.export_chrome_trace(str(output/f'baseline_history{history}.trace.json'))
                del prof;gc.collect()
            report['samples'].append(item);atomic_json(output/'profile_breakdown.json',report)
    report['passed']=True;atomic_json(output/'profile_breakdown.json',report)
    rows=[dict(history_tokens=item['history_tokens'],**row)for item in report['samples']for row in item.get('profile',{}).get('components',[])]
    if rows:
        with(output/'profile_breakdown.csv').open('w')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
