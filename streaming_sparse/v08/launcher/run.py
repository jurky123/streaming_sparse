"""Keep the original dependency graph and use equivalence-audited GPU workers."""
import argparse
import json
from pathlib import Path
import shutil
from ..common import HERE,atomic_json,source_identity
from .. import workflow
from .worker import require_execution_audit,launcher_identity
from ..execution import accelerate


def restrict_gpus(allowed):
    """Keep excluded physical GPUs unavailable even if they later become idle."""
    allowed=set(allowed)
    if not allowed or not allowed.issubset({0,1}):
        raise ValueError('Choose physical GPU 0 and/or 1')
    original_busy=workflow.gpu_busy
    workflow.gpu_busy=lambda:original_busy()|({0,1}-allowed)
    return sorted(allowed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through',choices=['prepare','pilot','matrix','final','test'],default='pilot')
    parser.add_argument('--gpus',type=int,nargs='+',choices=[0,1],help='Allowed physical GPUs; defaults to the saved selection')
    args=parser.parse_args();require_execution_audit()
    state_path=HERE/'reports/workflow/status.json'
    state=json.loads(state_path.read_text())if state_path.exists()else{}
    allowed=restrict_gpus(args.gpus if args.gpus is not None else state.get('allowed_gpus',[0,1]))
    original_plan=workflow.plan
    def plan():
        value=original_plan()
        for job in value['jobs']:
            if job['module'].rsplit('.',1)[-1]in('train','evaluate','rollout','warmstart'):
                job['args']=[job['module'],*job['args']]
                job['module']='streaming_sparse.v08.launcher.worker'
        value.update(execution_sha256=accelerate.identity(),launcher_sha256=launcher_identity())
        value.update(allowed_gpus=allowed,gpu_worker_limit=len(allowed))
        return value
    workflow.plan=plan
    target=HERE/'reports/launched_execution'/launcher_identity()
    target.mkdir(parents=True,exist_ok=True)
    for namespace in ('execution','launcher'):
        directory=target/namespace;directory.mkdir(exist_ok=True)
        for path in(HERE/namespace).glob('*.py'):shutil.copyfile(path,directory/path.name)
    shutil.copyfile(HERE/'reports/execution/benchmark.json',target/'benchmark.json')
    atomic_json(target/'identity.json',dict(source_sha256=source_identity(),
        execution_sha256=accelerate.identity(),launcher_sha256=launcher_identity(),allowed_gpus=allowed))
    if state:
        # Preserve the resource choice across explicit continuation commands.
        # Do not alter a running scheduler's state underneath its owner.
        if any(j.get('status')=='running'for j in state.get('jobs',{}).values()):
            raise RuntimeError('A scheduler already has running jobs; do not launch twice')
        state.update(allowed_gpus=allowed,gpu_worker_limit=len(allowed),launcher_sha256=launcher_identity())
        atomic_json(state_path,state)
    workflow.run(args.through)


if __name__=='__main__':main()
