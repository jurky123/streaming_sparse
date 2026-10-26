"""Run the original training loop with sample-parallel accumulation.

Only execution changes: partition the sampled inputs, SUM their
already batch-scaled gradients BEFORE clipping, shard validation, and designate
rank zero as the only checkpoint/log writer. Scientific source stays intact.
"""
import argparse
import ast
from datetime import datetime, timezone, timedelta
import inspect
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from migration.runtime import router_fast
from streaming_sparse.v08.common import HERE, atomic_json, sha, source_identity
from streaming_sparse.v08.execution import accelerate

rank = 0
world = 1
run_info = {}
_step_started = None
live_status = None
execution_identity = None


def install_checkpoint_io(train,execution):
    """One writer for ALL checkpoints; leave the native AdamW class intact.

    The legacy optimizer hook captures its own save closure, bypassing a later
    train.save wrapper. Do not install it in a distributed worker. Recovery is
    already saved by record_update after every complete optimizer update.
    """
    original_save=train.save;original_json=train.atomic_json
    def save(path,value):
        if rank!=0:return
        if isinstance(value,dict)and 'source_sha256'in value:
            value={**value,'execution':execution}
        return original_save(path,value)
    def write_json(path,value):
        if rank==0:return original_json(path,value)
    train.save=save;train.atomic_json=write_json


def status_update(**fields):
    if rank==0 and live_status is not None:
        old=json.loads(live_status.read_text())if live_status.exists()else{}
        old.pop('physical_gpu',None)
        old.update(fields,updated_at=datetime.now(timezone.utc).isoformat(),
            physical_gpus=os.environ.get('SPARSE_GPU_LIST'),world_size=world,
            worker_pid=os.getpid(),controller_pid=int(os.environ.get('SPARSE_LAUNCHER_PID','0')),
            command=run_info.get('command'),execution=run_info)
        atomic_json(live_status,old)


def reduce_update(totals, rparams, lparams):
    """One small gradient collective; no communication for frozen backbone."""
    params = rparams+lparams
    present = torch.tensor([p.grad is not None for p in params],device='cuda',dtype=torch.int32)
    flat = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in params])
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    dist.all_reduce(present,op=dist.ReduceOp.MAX)
    offset=0
    for p,used in zip(params,present.cpu().tolist()):
        p.grad=flat[offset:offset+p.numel()].view_as(p) if used else None
        offset+=p.numel()
    names=list(totals)
    values=torch.tensor([totals[k]for k in names],device='cuda',dtype=torch.float64)
    dist.all_reduce(values,op=dist.ReduceOp.SUM)
    totals.update(zip(names,values.cpu().tolist()))


@torch.no_grad()
def parallel_evaluate(network,ds,directory,keep=.2):
    from streaming_sparse.v08.evaluate import evaluate
    from streaming_sparse.v08.metrics import summarize
    # Contiguous shards retain the native trajectory/history cache locality.
    class View:
        split=ds.split
        def __init__(self):
            self.indices=list(range(len(ds)*rank//world,len(ds)*(rank+1)//world))
            self.rows=[ds.rows[i]for i in self.indices]
        def __len__(self):return len(self.indices)
        def load(self,i):return ds.load(self.indices[i])
        def fingerprint(self):return ds.fingerprint()
    shard=Path(directory)/f'rank{rank}'
    start=time.perf_counter()
    evaluate(network,View(),shard,keep)
    dist.barrier()
    result=[None]
    if rank==0:
        rows=[]
        for i in range(world):rows.extend(json.loads((Path(directory)/f'rank{i}/rows.json').read_text()))
        by_id={r['sample_id']:r for r in rows}
        assert len(by_id)==len(ds)==len(rows),'Validation coverage changed'
        rows=[by_id[r['sample_id']]for r in ds.rows]
        summary=summarize(rows)
        atomic_json(Path(directory)/'rows.json',rows)
        atomic_json(Path(directory)/'summary.json',dict(source_sha256=source_identity(),split=ds.split,
            keep=keep,dataset_fingerprint=ds.fingerprint(),model_version=network.model_version,
            router_version=network.router_version,**summary))
        atomic_json(Path(directory)/'progress.json',dict(completed=len(ds),total=len(ds),split=ds.split,world_size=world))
        atomic_json(Path(directory)/'parallel_timing.json',dict(seconds=time.perf_counter()-start,world_size=world))
        result[0]=summary
        print(json.dumps(dict(event='parallel_validation_complete',inputs=len(rows),seconds=time.perf_counter()-start)),flush=True)
    dist.broadcast_object_list(result,src=0)
    return result[0]


def update_started():
    global _step_started
    if run_info and router_fast.identity()!=run_info['runtime_sha256']:
        raise RuntimeError('Execution adapter changed while training was running')
    if execution_identity:
        from streaming_sparse.v08.launcher.worker import launcher_identity
        if accelerate.identity()!=execution_identity['kernel_sha256']or launcher_identity()!=execution_identity['launcher_sha256']:
            raise RuntimeError('Audited execution source changed while training was running')
    _step_started=time.perf_counter()


def record_update(item,directory):
    elapsed=time.perf_counter()-_step_started
    seconds=torch.tensor(elapsed,device='cuda',dtype=torch.float64)
    dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
    if rank==0:
        item['execution_timing']=dict(update_seconds=float(seconds),world_size=world,
            samples_per_update=run_info['global_batch'],router_cuda_graph=True)
        with (directory/'metrics.jsonl').open('a')as f:f.write(json.dumps(item)+'\n')
        # Router-only recovery is small; retain every completed update rather
        # than discarding up to nine updates when switching execution later.
        from streaming_sparse.v08 import train
        frame=inspect.currentframe().f_back
        try:
            assert frame.f_code is train.run.__code__
            train.save(directory/'recovery.pt',frame.f_locals['snapshot'](item['updates']))
            atomic_json(directory/'recovery.json',dict(updates=item['updates'],parallel=run_info,
                source_sha256=source_identity(),saved_at=datetime.now(timezone.utc).isoformat()))
        finally:del frame
        atomic_json(directory/'parallel_status.json',dict(status='training',updates=item['updates'],
            last_update_seconds=float(seconds),updated_at=datetime.now(timezone.utc).isoformat(),**run_info))
        print(json.dumps(dict(event='update',updates=item['updates'],seconds=float(seconds),losses=item['losses'])),flush=True)
        status_update(status='training',training_started=True,last_completed_update=item['updates'],
                      last_update_seconds=float(seconds))


def patch_loop(train):
    # Transform four execution sites, leaving objective, LR schedule, sampler,
    # clipping, parameter checks, snapshots, and stopping budget as original.
    tree=ast.parse(inspect.getsource(train.run))
    counts=dict(shard=0,reduce=0,log=0,start=0,resume_validation=0)
    class Transform(ast.NodeTransformer):
        def visit_For(self,node):
            self.generic_visit(node)
            if isinstance(node.target,ast.Name)and node.target.id=='index'and isinstance(node.iter,ast.Name)and node.iter.id=='examples':
                node.iter=ast.parse('examples[_parallel_rank::_parallel_world]',mode='eval').body
                counts['shard']+=1
            return node
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='norms':
                counts['reduce']+=1
                return [ast.parse('_reduce_update(totals,rparams,lparams)').body[0],node]
            return node
        def visit_Expr(self,node):
            if ast.unparse(node)=='optimizer.zero_grad(set_to_none=True)':
                counts['start']+=1
                return [ast.parse('_update_started()').body[0],node]
            return node
        def visit_With(self,node):
            if any("directory / 'metrics.jsonl'"in ast.unparse(item.context_expr)for item in node.items):
                counts['log']+=1
                return ast.parse('_record_update(item,directory)').body[0]
            return self.generic_visit(node)
        def visit_If(self,node):
            if ast.unparse(node.test)=='begin == 0':
                # The original entry repeats initial validation when resuming
                # an already validated update-zero checkpoint. Retain it.
                node.test=ast.parse('begin == 0 and not curve',mode='eval').body
                counts['resume_validation']+=1
            return self.generic_visit(node)
    tree=Transform().visit(tree);ast.fix_missing_locations(tree)
    assert counts==dict(shard=1,reduce=1,log=1,start=1,resume_validation=1),counts
    train.__dict__.update(_parallel_rank=rank,_parallel_world=world,_reduce_update=reduce_update,
                          _update_started=update_started,_record_update=record_update)
    compiled=compile(tree,'<audited sample-parallel original train.run>','exec')
    exec(compiled,train.__dict__)
    return ast.unparse(tree),counts


def main():
    global rank,world,run_info,live_status,execution_identity
    p=argparse.ArgumentParser();p.add_argument('--resume',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--parallel-audit',required=True)
    p.add_argument('--status')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--batch-amendment',help='Recorded user authorization required for batch 8')
    p.add_argument('--group',default='F-K');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-updates',type=int,help='Only for isolated disposable equivalence checks')
    p.add_argument('--audit-directory',help='Isolated outputs for max-updates checks')
    args=p.parse_args()
    live_status=Path(args.status)if args.status else None
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world in (1,2,4,8)
    assert args.batch_size in (4,8)and args.batch_size%world==0
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',init_method=os.environ.get('SPARSE_RENDEZVOUS'),
        rank=rank,world_size=world,timeout=timedelta(minutes=60))
    from streaming_sparse.v08.launcher.worker import require_execution_audit,launcher_identity
    from streaming_sparse.v08 import train
    audit=json.loads(Path(args.audit).read_text())
    assert audit['passed']and audit['source_sha256']==source_identity()
    # Gate/full-input audit hashes the optimizing adapter, not this orchestration.
    assert audit['router_adapter_sha256']==sha(Path(router_fast.__file__))
    parallel_audit=json.loads(Path(args.parallel_audit).read_text())
    assert parallel_audit['passed']and parallel_audit['source_sha256']==source_identity()
    assert parallel_audit['router_adapter_sha256']==sha(Path(router_fast.__file__))
    assert parallel_audit['world_size']==world
    assert parallel_audit.get('global_batch',4)==args.batch_size
    if args.batch_size!=4:
        assert args.batch_amendment,'Changing global batch requires recorded authorization'
        amendment=json.loads(Path(args.batch_amendment).read_text())
        assert amendment['user_instruction']=='允许batch8'and amendment['new_global_batch']==args.batch_size
        batch_protocol_base=train.protocol
        def effective_protocol():
            value=batch_protocol_base();value['main']['accumulation']=args.batch_size
            return value
        train.protocol=effective_protocol
    require_execution_audit();accelerate.install();router_fast.install()
    transformed,counts=patch_loop(train)
    run_info=dict(world_size=world,runtime_sha256=router_fast.identity(),
        command=__import__('sys').argv,
        router_adapter_sha256=sha(Path(router_fast.__file__)),source_sha256=source_identity(),
        parent_resume=str(Path(args.resume).resolve()),router_audit_sha256=sha(args.audit),
        parallel_audit_sha256=sha(args.parallel_audit),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        physical_gpus=os.environ.get('SPARSE_GPU_LIST'),
        global_batch=args.batch_size,batch_amendment=args.batch_amendment,
        effective_protocol_sha256=hashlib.sha256(json.dumps(train.protocol(),sort_keys=True).encode()).hexdigest(),
        gradient_reduction=f'SUM of objective/{args.batch_size}, before clipping, original sample order',
        rank_zero_pid=os.getpid()if rank==0 else None)
    execution=dict(kernel_sha256=accelerate.identity(),launcher_sha256=launcher_identity(),
        benchmark_sha256=sha(HERE/'reports/execution/benchmark.json'),parallel=run_info)
    execution_identity=execution
    install_checkpoint_io(train,execution)
    train.evaluate=parallel_evaluate
    if args.max_updates:
        assert args.audit_directory,'Never shorten the live experiment'
        original_protocol=train.protocol
        def protocol():
            value=original_protocol();value['main']['updates']=args.max_updates
            value['main']['validate_every']=args.max_updates
            return value
        train.protocol=protocol
        train.output=lambda path:Path(args.audit_directory).resolve()
        Path(args.audit_directory).mkdir(parents=True,exist_ok=True)
    if rank==0:
        root=Path(args.audit_directory)if args.audit_directory else HERE/f'runs/seed{args.seed}/{args.group}'
        atomic_json(root/'parallel_execution.json',dict(**run_info,patch_sites=counts,
            started_at=datetime.now(timezone.utc).isoformat()))
        (root/'parallel_train.generated.py').write_text(transformed+'\n')
        status_update(status='starting_multigpu',training_started=False)
    try:
        train.run('main',args.seed,group=args.group,device='cuda',resume=args.resume)
        status_update(status='completed',training_started=True,last_completed_update=train.protocol()['main']['updates'])
    except BaseException as error:
        status_update(status='multigpu_failed',error=repr(error));raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
