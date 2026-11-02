"""Resumable sequential driver; stdlib only; stages and errors remain visible."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('artifacts/rollout_acceptance_20261030')
def record(value):
    path=ROOT/'driver_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args()
    if a.wait_pid:
        while Path(f'/proc/{a.wait_pid}').exists():time.sleep(5)
    stages=[('quiet_control','rollout_quiet_control'),('suite','rollout_suite'),('profile','rollout_profile'),('secondary_quality','rollout_secondary_quality'),('analysis','rollout_analyze')]
    completed=[]
    for name,module in stages:
        with (ROOT/f'{name}.log').open('a') as log:
            log.write(f'\nDriver stage start unix_time={time.time()}\n');log.flush()
            child=subprocess.Popen(['bash','local_setup/run.sh','-m',f'local_experiments.{module}'],stdout=log,stderr=subprocess.STDOUT)
            record(dict(stage=name,child_pid=child.pid,completed=completed,driver_pid=os.getpid()))
            monitor=None
            if name not in ['quiet_control','analysis']:
                monitor=subprocess.Popen([sys.executable,'-m','local_experiments.rollout_monitor','--pid',str(child.pid),'--gpu',os.environ['CUDA_VISIBLE_DEVICES']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            code=child.wait()
            if monitor:monitor.wait(timeout=40)
            completed.append(dict(stage=name,returncode=code))
            record(dict(stage=name,returncode=code,completed=completed,driver_pid=os.getpid()))
            if code and name=='suite':return
    record(dict(stage='finished',completed=completed,all_stages_passed=all(x['returncode']==0 for x in completed)))
if __name__=='__main__':main()
