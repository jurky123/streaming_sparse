"""Start/resume the authorized eight-GPU branch from its newest checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PREP=ROOT/'migration/restoration/20261022_133350'
RUN=ROOT/'streaming_sparse/v08/runs/seed0/F-K'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args()
    status=PREP/'batch8_training/launcher_status.json'
    if status.exists()and not args.check_only:
        current=json.loads(status.read_text());pid=current.get('launcher_pid')
        process=Path(f'/proc/{pid}/cmdline')
        if current['status']=='running'and process.exists()and b'migration/runtime/launch.py'in process.read_bytes():
            print(json.dumps(dict(status='already_running',launcher_pid=pid,status_file=str(status))))
            return
    import torch
    from streaming_sparse.v08.common import source_identity
    choices=[]
    for name in ('recovery.pt','latest.pt','last.pt'):
        path=RUN/name
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert state['source_sha256']==source_identity()and state['group']=='F-K'and state['seed']==0
            if state['updates']>0:
                assert state['provenance']['settings']['accumulation']==8,'Different batch; inspect before resume'
            choices.append((state['updates'],path))
    step,resume=max(choices,key=lambda value:value[0])
    print(json.dumps(dict(status='resume_checked',checkpoint=str(resume),updates=step,global_batch=8)),flush=True)
    if args.check_only or step>=2000:return
    os.chdir(ROOT)
    command=[sys.executable,'-u',str(ROOT/'migration/runtime/launch.py'),
        '--gpus','0,1,2,3,4,5,6,7','--module','migration.runtime.multigpu',
        '--logdir',str(PREP/'batch8_training'),'--','--resume',str(resume),
        '--audit',str(PREP/'router_speed_audit.json'),'--parallel-audit',str(PREP/'parallel_batch8_audit.json'),
        '--batch-size','8','--batch-amendment',str(PREP/'batch8_amendment.json'),
        '--status',str(PREP/'continuation_status.json')]
    optimized=ROOT/'migration/routerslim_deployment.json'
    if optimized.exists():
        deployment=json.loads(optimized.read_text())
        from streaming_sparse.v08.common import sha
        audit=Path(deployment['audit']);assert sha(audit)==deployment['audit_sha256']
        verified=json.loads(audit.read_text());assert verified['passed']
        assert verified['checkpoint_sha256']==sha(deployment['baseline_checkpoint'])
        command[command.index('--module')+1]='migration.routerslim.train_module'
        command.extend(['--routerslim-audit',str(audit)])
    os.execv(sys.executable,command)


if __name__=='__main__':main()
