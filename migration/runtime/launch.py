"""Single-node process launcher with local-file rendezvous and per-rank logs."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--module',required=True)
    p.add_argument('--logdir',required=True);p.add_argument('arguments',nargs=argparse.REMAINDER)
    args=p.parse_args();gpus=args.gpus.split(',');world=len(gpus)
    assert world in (1,2,4,8)
    root=Path(__file__).resolve().parents[2];logs=Path(args.logdir).resolve();logs.mkdir(parents=True,exist_ok=True)
    lock=(logs/'launcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=uuid.uuid4().hex;rdzv=Path('/tmp')/f'streaming_sparse_{token}.rdzv'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,PYTHONPATH=str(root),
        SPARSE_RENDEZVOUS=f'file://{rdzv}',WORLD_SIZE=str(world),OMP_NUM_THREADS='4',
        SPARSE_GPU_LIST=args.gpus,SPARSE_LAUNCHER_PID=str(os.getpid()),
        OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
        NCCL_SOCKET_IFNAME='lo',NCCL_IB_DISABLE='1')
    # NVLink/P2P remains enabled. Inter-node RDMA is unnecessary on one node.
    arguments=args.arguments[1:]if args.arguments[:1]==['--']else args.arguments
    command=[sys.executable,'-u','-m','migration.runtime.bootstrap',args.module,*arguments]
    children=[];handles=[]
    state=dict(status='starting',launcher_pid=os.getpid(),gpus=gpus,world_size=world,
        rendezvous=str(rdzv),command=command,started_at=datetime.now(timezone.utc).isoformat())
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        state['ranks']=[dict(rank=i,pid=c.pid,returncode=c.poll())for i,c in enumerate(children)]
        tmp=logs/'launcher_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(logs/'launcher_status.json')
    def stop(signum,frame):
        for child in children:
            if child.poll()is None:child.terminate()
        state.update(status='stopped',signal=signum);save();raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for rank in range(world):
            rank_env=env.copy();rank_env.update(RANK=str(rank),LOCAL_RANK='0',CUDA_VISIBLE_DEVICES=gpus[rank])
            handle=(logs/f'rank{rank}.log').open('w');handles.append(handle)
            children.append(subprocess.Popen(command,cwd=root,env=rank_env,stdout=handle,stderr=subprocess.STDOUT))
        state['status']='running';save();print(json.dumps(state),flush=True)
        while any(c.poll()is None for c in children):
            failed=[c for c in children if c.poll()not in(None,0)]
            if failed:
                for c in children:
                    if c.poll()is None:c.terminate()
                state.update(status='failed',error='One rank failed; other ranks terminated');save()
                return 1
            save();time.sleep(5)
        state['status']='completed'if all(c.returncode==0 for c in children)else'failed';save()
        return 0 if state['status']=='completed'else 1
    finally:
        for h in handles:h.close()
        rdzv.unlink(missing_ok=True)


if __name__=='__main__':sys.exit(main())
