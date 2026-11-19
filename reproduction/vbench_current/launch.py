from pathlib import Path
import sqlite3,subprocess,os,time,sys
from concurrent.futures import ThreadPoolExecutor
from paths import RUN as r
from paths import REPO as repo, CODE
py=sys.executable
first=[('dense','short'),('lightforcing','long'),('base','short'),('intersection','short'),('step','short'),('lightforcing','short'),('intersection','long'),('step','long')]
def generate(g):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),PYTHONPATH=str(repo),OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false')
    db=sqlite3.connect(r/'jobs.sqlite',timeout=60)
    group=first[g]; failures=0
    with (r/'logs'/f'gen{g}.log').open('a') as log:
        while group:
            rc=subprocess.call([py,'-u',CODE/'worker.py',*group],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=repo)
            db.execute("update jobs set state=case when tries<2 then 'pending' else 'failed' end where state='running' and worker=?",(g,));db.commit()
            failures=failures+1 if rc else 0
            if failures>=3:raise RuntimeError(f'GPU {g}: {failures} consecutive worker failures')
            group=db.execute("select method,kind from jobs where state='pending' group by method,kind order by count(*)*(case kind when 'long' then 5 else 1 end) desc limit 1").fetchone()
def score(g):
    with (r/'logs'/f'score{g}.log').open('a') as f:
        subprocess.run([sys.executable,'-u',CODE/'score_shard.py'],env=dict(os.environ,SCORE_GPU=str(g)),stdout=f,stderr=subprocess.STDOUT,check=True)
with ThreadPoolExecutor(8) as pool:
    list(pool.map(generate,range(8)))
    db=sqlite3.connect(r/'jobs.sqlite')
    if db.execute("select count(*) from jobs where state!='done'").fetchone()[0]:raise RuntimeError('unfinished generation')
    list(pool.map(score,range(8)))
(r/'COMPLETE').touch()
