"""Bounded resumable stages; completion requires committed result counts."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('artifacts/compensation_probe_20261030')
def write(value):
    p=ROOT/'driver_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,p)
def complete(stage):
    if stage=='compensation_select':return (ROOT/'selection.json').exists()
    if stage=='compensation_kernel_checks':return (ROOT/'kernel_checks.json').exists() and json.loads((ROOT/'kernel_checks.json').read_text())['passed']
    if stage=='compensation_deployment_checks':return (ROOT/'deployment/gate.json').exists() and json.loads((ROOT/'deployment/gate.json').read_text())['passed']
    if stage=='compensation_accept':return (ROOT/'acceptance_summary.json').exists() and json.loads((ROOT/'acceptance_summary.json').read_text())['complete']
    return False

def main():
    ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'driver.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'status.json').read_text())['stage']=='complete','Finish resumable probe first'
    for stage in ['compensation_select','compensation_kernel_checks','compensation_deployment_checks','compensation_accept']:
        if complete(stage):continue
        with (ROOT/f'{stage}.log').open('a') as log:
            child=subprocess.Popen(['bash','local_setup/run.sh','-m','local_experiments.'+stage],stdout=log,stderr=subprocess.STDOUT)
            write(dict(stage=stage,driver_pid=os.getpid(),child_pid=child.pid));code=child.wait()
        if code or not complete(stage):write(dict(stage='failed' if code else 'interrupted',task=stage,returncode=code));return
    write(dict(stage='complete',note='Execution complete; quality/speed acceptance remains a scientific judgement.'))
if __name__=='__main__':main()
