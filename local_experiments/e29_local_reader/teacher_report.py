from .core import *
from ..e25_decision_diagnostics.report import aggregate
import csv

def table(name,rows):
 if not rows:return
 with (E29/name).open('w') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 rows=[];summary=[];replay=[]
 for seed in [17,29]:
  dest=E29/'teacher'/f's{seed}';assert (dest/'complete.json').exists();rr=json.loads((dest/'decisions.json').read_text());rows+=rr;old=json.loads((ROOT/'artifacts/e28_local_response_v1'/f's{seed}'/'decisions.json').read_text());old={(x['event'],x['policy']):x for x in old if x['method']=='Local-LOTO'}
  for row in rr:
   if row['method']=='FullTeacher':
    ref=old[row['event'],row['policy']];assert (row['remove'],row['add'],row['execute'])==(ref['remove'],ref['add'],ref['execute']);assert row['executed_gain']==ref['executed_gain']
  replay.append(dict(seed=seed,fullteacher_exact_replay=True,events=288))
  for method in ['FullTeacher','HistoryOnly','CurrentOnly']:
   for policy in ['Set','Independent']:summary.append(dict(seed=seed,method=method,policy=policy,**aggregate([r for r in rr if r['method']==method and r['policy']==policy])))
 table('teacher_per_event.csv',rows);table('teacher_summary.csv',summary);dump(E29/'teacher_replay.json',replay);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
