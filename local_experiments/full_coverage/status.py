"""Lightweight status/ETA reader; no model loading or new tests."""
import argparse,datetime,json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/full_coverage_v1'

def collect():
    rows=[]
    for seed in (17,29):
        folder=OUT/f'training/C_s{seed}';s=json.loads((folder/'status.json').read_text());old=[]
        for line in (folder/'run.log').read_text().splitlines():
            m=re.fullmatch(fr'{seed} (\d+) 348 (\d+)',line)
            if m:old.append(tuple(map(int,m.groups())))
        old=old[-24:];old_rate=(old[-1][1]-old[0][1])/(old[-1][0]-old[0][0]) if len(old)>1 else None
        u=s.get('update',s.get('updates',0));total=s.get('total',66816);rate=s.get('seconds_per_step_recent')
        pid=s.get('pid');proc=Path('/proc')/str(pid)
        try:alive=proc.exists() and proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
        except FileNotFoundError:alive=False
        rows.append(dict(seed=seed,status=s['status'],update=u,total=total,percent=100*u/total,positions=s.get('positions'),pid=pid,alive=alive,step_seconds=rate,legacy_step_seconds=old_rate,
                         speedup=old_rate/rate if old_rate and rate else None,remaining_train_minutes=max(0,total-u)*rate/60 if rate else None,numeric_repaired_events=s.get('numeric_repaired_events',0),evaluation_pending=s.get('evaluation_pending',[])))
    remaining=max((r['remaining_train_minutes'] or 0) for r in rows)
    healthy=all(r['status']=='complete' or (r['alive'] and r['status'] in ('training','validation')) for r in rows)
    finished=all(r['status']=='complete' for r in rows)
    return dict(eta_valid=healthy,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='current full-coverage C joint training only; excludes later Full_Pset adaptation',seeds=rows,remaining_train_minutes=remaining if healthy else None,
                projected_total_minutes=([0,0] if finished else [remaining+5,remaining+15]) if healthy else None,eta_note='recent wall-clock rate includes checkpoint validation/IO; add 5-15 minutes for final sharded evaluation/export, assuming no new errors',
                training_gpus=[0,1],validation_gpus=[2,3,4,5,6,7],baseline_cache_ready=(OUT/'execution_optimization/old_validation/baseline.json').exists())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');a=p.parse_args();value=collect();text=json.dumps(value,indent=2,ensure_ascii=False)
    if a.save:(OUT/'execution_optimization/current_status.json').write_text(text+'\n')
    print(text)
