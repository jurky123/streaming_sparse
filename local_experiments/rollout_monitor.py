"""Bounded GPU telemetry; exits when the supplied worker PID exits."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--gpu',default='5');a=p.parse_args()
out=Path('artifacts/rollout_acceptance_20261030/gpu_telemetry.jsonl')
while Path(f'/proc/{a.pid}').exists():
    data={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_pid':a.pid}
    for key,cmd in [('gpu',['nvidia-smi','-i',a.gpu,'--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.free,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader']),('processes',['nvidia-smi','-i',a.gpu,'--query-compute-apps=pid,used_memory','--format=csv,noheader'])]:
        try:data[key]=subprocess.check_output(cmd,text=True,timeout=10).strip()
        except Exception as exc:data[key]=repr(exc)
    with out.open('a') as f:f.write(json.dumps(data)+'\n')
    time.sleep(15)
