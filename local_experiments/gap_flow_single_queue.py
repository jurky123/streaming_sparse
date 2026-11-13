"""Independent resumable subprocess queue; never modifies the main worker plan."""
import os,subprocess,sys,time
from pathlib import Path
from .learned_cache.io import RunLock,atomic
ROOT=Path('artifacts/flow_single_20261104')
def main():
    lock=RunLock(ROOT/'worker')
    jobs=[('train_city','gap_flow_single_train',['--case','city_walk','--initial','artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt']),('train_reef','gap_flow_single_train',['--case','reef_fish','--initial',str(ROOT/'train/city_walk/model.pt')])]
    jobs += [('video_'+c,'gap_flow_single_video',['--case',c,'--seed','101','--variant','full','--chunks','9']) for c in ['horse_field','traffic_crossing']]
    jobs += [('report','gap_flow_single_report',[])]
    try:
        for name,module,args in jobs:
            if (ROOT/'worker'/f'{name}.done.json').exists():continue
            with (ROOT/'worker'/f'{name}.log').open('a') as log:
                p=subprocess.Popen([sys.executable,'-u','-m','local_experiments.'+module]+args,stdout=log,stderr=subprocess.STDOUT)
                atomic(ROOT/'worker/status.json',dict(status='running',job=name,pid=os.getpid(),child_pid=p.pid,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),updated=time.time(),speed_valid=False))
                code=p.wait()
            if code:raise RuntimeError(f'{name} exited {code}')
            atomic(ROOT/'worker'/f'{name}.done.json',dict(status='complete'))
        atomic(ROOT/'worker/status.json',dict(status='complete',speed_valid=False))
    except BaseException as e:atomic(ROOT/'worker/status.json',dict(status='failed',error=repr(e)));raise
    finally:lock.close()
if __name__=='__main__':main()
