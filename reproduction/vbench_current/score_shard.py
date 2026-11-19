from resources import ready
import hashlib,json,os,sqlite3,subprocess,time
from pathlib import Path
from paths import RUN as r

prompt={(x['kind'],x['i']):x for x in json.loads((r/'prompts.json').read_text())}
longdims='imaging_quality dynamic_degree motion_smoothness subject_consistency background_consistency aesthetic_quality overall_consistency chunk_discontinuity'.split()
dims=list(dict.fromkeys(['imaging_quality','temporal_flickering']+[d for x in prompt.values() for d in x['dimensions']]))
db=sqlite3.connect(r/'jobs.sqlite',timeout=40)
db.execute('create table if not exists scores (jobid integer, dim text, batch text, primary key(jobid,dim))')
last={};start=time.time()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=os.getenv('SCORE_GPU','6'),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='3',OMP_NUM_THREADS='3',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(r/'eval_deps')+':'+str(r/'VBench-master'),VBENCH_CACHE_DIR=str(r/'cache'),TORCH_HOME=str(r/'torch_cache'))
from paths import CODE
py=os.environ.get('VBENCH_SCORE_PYTHON',str(r/'eval_env_local/bin/python'))
while True:
    tasks=db.execute("select rowid,kind,i,seed,method from jobs where state='done' order by priority,rowid").fetchall()
    done=set(db.execute('select jobid,dim from scores'))
    options=[]
    for k,ds in [('short',dims),('long',longdims)]:
        for d in ds:
            if not ready(d):continue
            ids=[t for t in tasks if t[0]%8==int(env["CUDA_VISIBLE_DEVICES"]) and t[1]==k and (t[0],d) not in done and (k=='long' or d in prompt[(k,t[2])]['dimensions'])]
            if ids and time.time()-last.get((k,d),0)>0:options.append((last.get((k,d),0),k,d,ids))
    if not options:
        expected=sum(len(longdims) if t[1]=='long' else len(set(prompt[(t[1],t[2])]['dimensions'])) for t in tasks if t[0]%8==int(env['CUDA_VISIBLE_DEVICES']))
        if len(done.intersection({(t[0],d) for t in tasks if t[0]%8==int(env['CUDA_VISIBLE_DEVICES']) for d in (longdims if t[1]=='long' else prompt[(t[1],t[2])]['dimensions'])}))==expected:break
        time.sleep(6);continue
    _,k,d,ids=min(options,key=lambda o:o[0]);ids=ids[:(48 if k=='short' else 6)]
    name=f'{k}_{d}_'+hashlib.sha256(str([t[0] for t in ids]).encode()).hexdigest()[:14]
    b=r/'metrics'/name;b.mkdir(exist_ok=True)
    v=[dict(i=i,method=m,seed=s,path=str(r/'videos'/m/k/f'{i}_{s}.mp4'),prompt=prompt[(k,i)]['prompt']) for _,k,i,s,m in ids]
    (b/'jobs.json').write_text(json.dumps(dict(kind=k,dimension=d,videos=v)))
    with (b/'run.log').open('a') as log:
        try:
            if not (b/'result.json').exists():subprocess.run([py,'-u',CODE/'score_batch.py',b],stdout=log,stderr=subprocess.STDOUT,timeout=720,env=env,cwd=r/'VBench-master',check=True)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
            last[(k,d)]=time.time()+300
            print('retry',name,flush=True);continue
    db.executemany('insert or ignore into scores values(?,?,?)',[(t[0],d,name) for t in ids]);db.commit()
    last[(k,d)]=time.time()
    print('done',name,len(ids),flush=True)
