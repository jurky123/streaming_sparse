"""Opt in to paired F-K/F-V training in an isolated experiment directory."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import textwrap
from datetime import timedelta
import torch.distributed as dist
from migration.paired_router import runtime
from migration.runtime import multigpu
from migration.routerslim import r1
from streaming_sparse.v08.common import sha,source_identity


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--directory',required=True);p.add_argument('--stage',required=True)
    ours,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    group=remaining[remaining.index('--group')+1];audit_path=remaining[remaining.index('--parallel-audit')+1]
    audit=json.loads(Path(audit_path).read_text());assert audit['passed']and audit['group']==group and audit['stage']==ours.stage
    assert audit['paired_runtime_sha256']==runtime.identity()and audit['r1_sha256']==r1.identity()
    manifest,spec=runtime.install_training(ours.directory,ours.stage,group)
    multigpu.HERE=Path(spec['directory'])
    assert audit['layers']==spec['layers']
    entry_sha=sha(__file__);runtime_sha=runtime.identity();r1_sha=r1.identity()
    multigpu.accelerate.install();multigpu.router_fast.install();r1.install(variance=True,projection=False,skip_zero_aux=True)
    old_start=multigpu.update_started
    def start():
        assert sha(__file__)==entry_sha and runtime.identity()==runtime_sha and r1.identity()==r1_sha,'Paired execution changed while running'
        if(Path(ours.directory)/'STOP').exists():raise RuntimeError('User stop request; last complete checkpoint retained')
        frame=inspect.currentframe().f_back
        try:r1.configure_update(frame.f_locals['network'],frame.f_locals['step'])
        finally:del frame
        r1.stats.update(boundary_groups=0,score_queries=0);old_start()
    multigpu.update_started=start
    original_io=multigpu.install_checkpoint_io
    def install_io(train,execution):
        multigpu.run_info['paired_training']=dict(experiment=str(Path(ours.directory).resolve()),stage=ours.stage,
            layers=spec['layers'],entry_sha256=entry_sha,paired_runtime_sha256=runtime_sha,r1_sha256=r1_sha,
            backbone_frozen=True,lora_frozen=True,train_fingerprint=manifest['train_fingerprint'],
            min_updates=spec['min_updates'],max_updates=spec['max_updates'],convergence=manifest['convergence'])
        return original_io(train,execution)
    multigpu.install_checkpoint_io=install_io
    source=textwrap.dedent(inspect.getsource(multigpu.record_update))
    old='    elapsed=time.perf_counter()-_step_started';assert source.count(old)==1
    source=source.replace(old,"""    item['att_aux_computed']=item['updates']<501
    if not item['att_aux_computed']:item['losses']['att_aux']=None
    item['paired_training']=run_info['paired_training']
    item['routerslim_boundary_counts_rank0']=dict(_paired_r1.stats)
    elapsed=time.perf_counter()-_step_started""")
    multigpu.__dict__['_paired_r1']=r1;exec(compile(source,'<paired metrics metadata>','exec'),multigpu.__dict__)
    original_init=dist.init_process_group
    def init(*args,**kwargs):
        kwargs['timeout']=timedelta(hours=8)
        return original_init(*args,**kwargs)
    dist.init_process_group=init
    multigpu.main()


if __name__=='__main__':main()
