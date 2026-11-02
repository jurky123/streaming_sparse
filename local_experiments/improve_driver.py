"""Stdlib stage driver. Each experiment owns its lock/checkpoints; never skips failed gates."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('artifacts/rollout_improve_20261030')
def save(value):
    p=ROOT/'driver_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args()
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(dict(stage='waiting_for_training',driver_pid=os.getpid(),wait_pid=a.wait_pid))
    if a.wait_pid:
        while Path(f'/proc/{a.wait_pid}').exists():time.sleep(5)
    stages=['improve_train','improve_resume_real','improve_select','improve_bench','improve_score_checks','improve_eval_checks','improve_accept','improve_report'];done=[]
    for module in stages:
        with (ROOT/f'{module}.log').open('a') as log:
            child=subprocess.Popen(['bash','local_setup/run.sh','-m',f'local_experiments.{module}'],stdout=log,stderr=subprocess.STDOUT)
            save(dict(stage=module,driver_pid=os.getpid(),child_pid=child.pid,completed=done));code=child.wait();done.append(dict(stage=module,returncode=code))
        if code:
            save(dict(stage='failed',failed_stage=module,returncode=code,driver_pid=os.getpid(),completed=done));return
    save(dict(stage='finished',completed=done,note='Pipeline completion is not a joint quality/speed pass.'))
if __name__=='__main__':main()
