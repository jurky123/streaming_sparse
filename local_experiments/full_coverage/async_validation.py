"""Evaluate immutable epoch weights on spare GPUs while the next epoch trains."""
import collections,json,os,subprocess,sys
from pathlib import Path
from .common import ROOT,OUT,atomic,digest
from .evaluate_parallel import CACHE

class Validation:
    def __init__(self,root,seed,gpus,checks):
        self.root=Path(root);self.seed=seed;self.gpus=gpus;self.checks=list(checks);self.children={};self.all_paths=None
    def epochs(self):return [e for e in (0,1) if (self.root/f'epoch{e}_weights.pt').exists() and e not in {c['epoch'] for c in self.checks}]
    def pending(self):return self.epochs()
    def poll(self):
        if not (CACHE/'baseline.json').exists():return
        for epoch in self.epochs():
            checkpoint=self.root/f'epoch{epoch}_weights.pt';dest=self.root/f'evaluation_epoch{epoch}';dest.mkdir(parents=True,exist_ok=True)
            source=dict(weights=digest(checkpoint),baseline=digest(CACHE/'baseline.json'));shards=len(self.gpus)
            if all((dest/f'part{i}.json').exists() for i in range(shards)):
                records=[]
                for i in range(shards):
                    rec=json.loads((dest/f'part{i}.json').read_text());assert rec['source']==source;records.extend(rec['rows'])
                records.sort(key=lambda r:r['event']);counts=collections.Counter(r['key'] for r in records)
                assert len(records)==8352 and len({r['event'] for r in records})==8352 and len(counts)==348 and set(counts.values())=={24}
                atomic(self.root/f'dev_epoch{epoch}.json',records);ratio=sum(r['initial'] for r in records)/sum(r['old'] for r in records)
                self.checks.append(dict(epoch=epoch,initial_ratio=ratio));self.checks.sort(key=lambda c:c['epoch'])
                import torch
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False);saved['checks']=list(self.checks);atomic(self.root/f'epoch{epoch}.pt',saved,True)
                continue
            for i,gpu in enumerate(self.gpus):
                output=dest/f'part{i}.json';key=(epoch,i)
                if output.exists():continue
                if key in self.children:
                    code=self.children[key].poll()
                    if code is None:continue
                    if code:raise RuntimeError(f'Evaluation failed: {dest}/part{i}.log')
                launch=dest/f'launch{i}.json';previous=json.loads(launch.read_text()) if launch.exists() else {}
                pid=previous.get('pid');proc=Path('/proc')/str(pid)/'cmdline'
                if proc.exists() and str(checkpoint).encode() in proc.read_bytes():continue
                if previous.get('attempts',0)>=3:raise RuntimeError(f'Evaluation repeatedly failed: {output}')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                cmd=[sys.executable,'-u','-m','local_experiments.full_coverage.evaluate_parallel','--mode','checkpoint','--checkpoint',str(checkpoint),'--seed',str(self.seed),'--shard',str(i),'--shards',str(shards),'--output',str(output)]
                with (dest/f'part{i}.log').open('a') as log:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                self.children[key]=p;atomic(launch,dict(pid=p.pid,gpu=gpu,command=cmd,attempts=previous.get('attempts',0)+1))
            # At most one epoch in flight per seed/GPU group.
            break
