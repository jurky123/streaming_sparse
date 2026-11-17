"""SQLite job claiming, immutable manifest, same-host orphan recovery."""
import hashlib,json,os,socket,sqlite3,time
from pathlib import Path
from ..learned_cache.io import atomic,digest
METHODS=('dense','c_risk','c_pset')

class Queue:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'jobs.sqlite',timeout=60,isolation_level=None)
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, spec TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL DEFAULT "pending", worker TEXT, updated REAL, error TEXT, UNIQUE(spec,stage))')
    def initialize(self,manifest):
        jobs=json.loads(Path(manifest).read_text());assert isinstance(jobs,list) and jobs
        rows=[]
        for job in jobs:
            if set(job)!={'case','noise_seed','method','chunks'} or job['method'] not in METHODS or job['chunks'] not in (9,41):raise ValueError(job)
            if not isinstance(job['noise_seed'],int) or not isinstance(job['case'],str) or not job['case'].replace('_','').isalnum():raise ValueError(job)
            canonical=json.dumps(job,sort_keys=True);key=hashlib.sha256(canonical.encode()).hexdigest()[:20];rows.append((key,canonical))
        pairs={(j['case'],j['noise_seed'],j['chunks'],j['method']) for j in jobs}
        for j in jobs:
            if (j['case'],j['noise_seed'],j['chunks'],'dense') not in pairs:raise ValueError('Manifest must include a Dense reference for every case/seed/length')
        existing=self.root/'manifest.json'
        if existing.exists():assert json.loads(existing.read_text())==jobs,'Manifest is immutable; choose a new run directory'
        else:atomic(existing,jobs)
        with self.transaction():
            for key,spec in rows:
                for stage in ('generate','postprocess'):self.db.execute('INSERT OR IGNORE INTO jobs (key,spec,stage) VALUES (?,?,?)',(key+'_'+stage,spec,stage))
    def bind_identity(self,identity):
        canonical=json.dumps(identity,sort_keys=True)
        with self.transaction():
            row=self.db.execute('SELECT value FROM metadata WHERE name="execution"').fetchone()
            if row is not None:
                if row[0]!=canonical:raise ValueError('Execution identity changed; use a new run directory')
            else:
                self.db.execute('INSERT INTO metadata VALUES (?,?)',('execution',canonical))
                atomic(self.root/'execution_identity.json',identity)
    def transaction(self):
        from contextlib import contextmanager
        @contextmanager
        def tx():
            self.db.execute('BEGIN IMMEDIATE')
            try:yield;self.db.execute('COMMIT')
            except BaseException:self.db.execute('ROLLBACK');raise
        return tx()
    def recover(self):
        host=socket.gethostname()
        with self.transaction():
            for key,worker in self.db.execute('SELECT key,worker FROM jobs WHERE status="running"').fetchall():
                whost,pid=worker.split(':')
                if whost==host and not Path('/proc',pid).exists():self.db.execute('UPDATE jobs SET status="pending",worker=NULL WHERE key=?',(key,))
    def claim(self,stage,worker):
        with self.transaction():
            if stage=='postprocess':
                # Process Dense references first; sparse quality requires their decode cache.
                rows=self.db.execute('SELECT key,spec FROM jobs WHERE stage=? AND status="pending" ORDER BY json_extract(spec,"$.method")!="dense",key',(stage,)).fetchall()
                row=None
                for key,spec in rows:
                    j=json.loads(spec);gen=self.db.execute('SELECT status FROM jobs WHERE spec=? AND stage="generate"',(spec,)).fetchone()
                    if not gen or gen[0]!='done':continue
                    ref=dict(j,method='dense');rs=json.dumps(ref,sort_keys=True)
                    if j['method']!='dense':
                        status=self.db.execute('SELECT status FROM jobs WHERE spec=? AND stage="postprocess"',(rs,)).fetchone()
                        if not status or status[0]!='done':continue
                    row=(key,spec);break
            else:row=self.db.execute('SELECT key,spec FROM jobs WHERE stage=? AND status="pending" ORDER BY json_extract(spec,"$.method")!="dense",key LIMIT 1',(stage,)).fetchone()
            if row is None:return None
            key,spec=row;self.db.execute('UPDATE jobs SET status="running",worker=?,updated=?,error=NULL WHERE key=?',(worker,time.time(),key))
            return key,json.loads(spec)
    def update(self,key,worker,status,error=None):
        cur=self.db.execute('UPDATE jobs SET status=?,updated=?,error=? WHERE key=? AND worker=?',(status,time.time(),error,key,worker))
        if cur.rowcount!=1:raise RuntimeError('Job ownership changed')
    def status(self):return [dict(stage=s,status=t,count=n) for s,t,n in self.db.execute('SELECT stage,status,count(*) FROM jobs GROUP BY stage,status')]

def folder(root,job):return Path(root)/'videos'/f'{job["case"]}_s{job["noise_seed"]}'/f'chunks{job["chunks"]}'/job['method']

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    q=Queue(a.root);q.recover()
    if a.retry_failed:q.db.execute('UPDATE jobs SET status="pending",worker=NULL WHERE status="failed"')
    print(json.dumps(q.status()))
