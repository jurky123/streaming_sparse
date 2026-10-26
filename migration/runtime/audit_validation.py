"""Verify sharded full validation against the saved single-GPU validation."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from migration.runtime import multigpu,router_fast
from streaming_sparse.v08.common import atomic_json,paired_seed,sha,source_identity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--reference',required=True);parser.add_argument('--directory',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',init_method=os.environ['SPARSE_RENDEZVOUS'],rank=rank,world_size=world,
        timeout=timedelta(minutes=60))
    paired_seed(0);multigpu.rank=rank;multigpu.world=world
    from streaming_sparse.v08.execution import accelerate
    from streaming_sparse.v08.evaluate import load_network
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.capture.capture_runner import sdpa_backend
    accelerate.install();router_fast.install()
    network,state=load_network(args.checkpoint)
    ds=ReplayDataset('val',require_complete=True)
    report=dict(passed=False,source_sha256=source_identity(),world_size=world,
        router_adapter_sha256=sha(Path(router_fast.__file__)))
    try:
        with sdpa_backend():metrics=multigpu.parallel_evaluate(network,ds,args.directory,.2)
        if rank==0:
            reference=json.loads(Path(args.reference).read_text());actual=json.loads((Path(args.directory)/'rows.json').read_text())
            assert [r['sample_id']for r in actual]==[r['sample_id']for r in reference]
            differences=[abs(a['nmse']-b['nmse'])for a,b in zip(actual,reference)]
            assert max(differences)<=5e-5
            report.update(passed=True,inputs=len(actual),maximum_input_nmse_difference=max(differences),
                exact_sse_inputs=sum(a['sse']==b['sse']for a,b in zip(actual,reference)),
                reference_prompt_equal_error=sum(r['nmse']for r in reference)/len(reference),
                parallel_prompt_equal_error=metrics['prompt_equal_error'],
                coverage_and_order_exact=True,test_split_accessed=False)
            atomic_json(args.output,report);print(json.dumps(report),flush=True)
        dist.barrier()
    except BaseException as error:
        if rank==0:report['error']=repr(error);atomic_json(args.output,report)
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
