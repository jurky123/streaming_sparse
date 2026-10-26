"""Seal implementation, run clean paired speed trials, then all Val videos."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from ..common import ROOT,PYTHON,config,atomic_json,update,sha


VARIANTS={name:dict(method='streamgram',precision='fp16',score_precision=sp,gp=8,tile=64,batch=512)
          for name,sp in [('Hybrid','tf32x3'),('FP16','fp16')]}
ASSIGNMENTS=[('Dense','Dense','reference'),('Opt2-K32','K32-Set','reference'),('Opt2-V32','V32-Set','reference'),
             ('Hybrid-K32','K32-Set','Hybrid'),('Hybrid-V32','V32-Set','Hybrid'),
             ('FP16-K32','K32-Set','FP16'),('FP16-V32','V32-Set','FP16'),('Hybrid-K32-replica','K32-Set','Hybrid')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root,_=config(a.directory);out=root/'optimization4';out.mkdir(exist_ok=True)
    if a.worker:
        name,mode,variant=ASSIGNMENTS[int(os.environ['RANK'])]
        from .. import fast_bench
        if variant=='reference':
            from ..opt2_guarded import load_guarded
            fast_bench.load_fast=load_guarded
        else:
            from .network import load_compact
            options={k:v for k,v in VARIANTS[variant].items()if k!='batch'}
            fast_bench.load_fast=lambda *a,**kw:load_compact(*a,**options,**kw)
        sys.argv=[sys.argv[0],'--directory',str(root),'--output',str(out/'benchmark'/name),'--mode',mode]
        fast_bench.main();return
    checks=json.loads((out/'checks.json').read_text());assert checks['status']=='completed'and checks['finite_nonempty']
    prior=json.loads((root/'optimization2/manifest.json').read_text())
    guarded=json.loads((root/'optimization2/guarded_manifest.json').read_text())
    old={**prior['code'],**prior['prior_code'],**prior['original_code'],**guarded['code'],
         **json.loads((root/'e2e_manifest.json').read_text())['frozen_files']}
    for f,digest in old.items():assert sha(f)==digest,f
    files=list(Path(__file__).parent.glob('*.py'))
    manifest=dict(code={str(f.resolve()):sha(f)for f in files},prior_unchanged=old,variants=VARIANTS,
        assignments=ASSIGNMENTS,query_stride=1,global_tf32=False,
        local_precision='Hybrid: TF32x3 score, FP16 Gram/mean/MLP; FP16: FP16 score and same statistics; all accumulated in FP32',
        quality_scope='All8 registered Val prompts x2 heads x2 new variants, seed0; Test sealed; no retraining')
    atomic_json(out/'manifest.json',manifest)
    for stage,module,extra in [('benchmark','migration.unified_router.opt4.suite',['--worker']),('quality','migration.unified_router.opt4.quality',[])]:
        update(out/'status.json',status='running',stage=stage,pid=os.getpid());print(stage,flush=True)
        cmd=[str(PYTHON),'-u',str(ROOT/'migration/runtime/launch.py'),'--gpus','0,1,2,3,4,5,6,7','--module',module,
            '--logdir',str(out/f'{stage}_logs'),'--','--directory',str(root),*extra]
        with (out/f'{stage}_launcher.log').open('w')as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:update(out/'status.json',status='failed',stage=stage);raise RuntimeError(stage)
    for f,digest in {**manifest['code'],**old}.items():assert sha(f)==digest,f
    update(out/'status.json',status='completed',stage='benchmark_and_quality_complete',sources_immutable=True)


if __name__=='__main__':main()
