import sqlite3,time
from pathlib import Path
from datetime import datetime,timezone
from paths import RUN as r
db=sqlite3.connect(f'file:{r}/jobs.sqlite?mode=ro',uri=True)
for row in db.execute("select method,kind,sum(state='done'),sum(state='running'),sum(state='failed'),count(*),round(avg(gen),2),round(avg(decode),2) from jobs group by method,kind"):
 print('%-12s %-5s done=%d running=%d failed=%d target=%d gen_s=%s vae_s=%s'%row)
for row in db.execute('select j.kind,s.dim,count(*) from scores s join jobs j on j.rowid=s.jobid group by 1,2 order by 1,2'):print('scored',*row)

# Each shard owns a disjoint set of video IDs. A worker may be loading a model.
import json, os, subprocess
prompts=json.loads((r/'prompts.json').read_text())
dims={(p['kind'],p['i']):len(p.get('dimensions',[])) for p in prompts}
targets={'short':0,'long':0}
for kind,i,n in db.execute('select kind,i,count(*) from jobs group by kind,i'):
 targets[kind]+=n*(8 if kind=='long' else dims[kind,i])
counts=dict(db.execute('select j.kind,count(*) from scores s join jobs j on j.rowid=s.jobid group by 1'))
print('Score progress:',', '.join(f'{k} {counts.get(k,0)}/{v}' for k,v in targets.items()))
