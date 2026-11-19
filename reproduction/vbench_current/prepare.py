"""Initialize the frozen workload in an empty run directory."""
import csv
import hashlib
import json
import sqlite3
from shutil import copyfile
from paths import CODE, REPO, RUN

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    for name,value in json.loads((CODE/'files.json').read_text()).items():
        if sha(REPO/name)!=value: raise ValueError(f'identity mismatch: {name}')
    RUN.mkdir(parents=True,exist_ok=True)
    dbpath=RUN/'jobs.sqlite'
    if dbpath.exists(): raise FileExistsError(dbpath)
    for name in ('logs','videos','embeddings','metrics','segments'): (RUN/name).mkdir(exist_ok=True)
    for name in ('info.json','prompts.json'):copyfile(CODE/name,RUN/name)
    with sqlite3.connect(dbpath) as db:
        db.execute('pragma journal_mode=wal')
        db.execute("create table jobs(kind text,i int,seed int,method text,priority int,state text default 'pending',worker int,tries int default 0,started real,gen real,decode real,error text,primary key(kind,i,seed,method))")
        db.execute("create table scores(jobid integer,dim text,batch text,primary key(jobid,dim))")
        with (CODE/'jobs.tsv').open() as f:
            db.executemany('insert into jobs(rowid,kind,i,seed,method,priority) values (?,?,?,?,?,?)',csv.reader(f,delimiter='\t'))
        assert db.execute('select count(*) from jobs').fetchone()==(5760,)
    print(f'{RUN}: 5760 jobs')
