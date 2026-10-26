"""Final selected-weight fusion: clean timings followed by bitwise regression."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .common import ROOT,PYTHON,config,atomic_json,update,sha
from .opt4.suite import VARIANTS,ASSIGNMENTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root,_=config(a.directory);out=root/'optimization4'
    if a.worker:
        name,mode,variant=ASSIGNMENTS[int(os.environ['RANK'])]
        from . import fast_bench
        if variant=='reference':
            from .opt2_guarded import load_guarded
            fast_bench.load_fast=load_guarded
        else:
            from .opt4_fused import load_fused
            options={k:v for k,v in VARIANTS[variant].items()if k!='batch'}
            fast_bench.load_fast=lambda *a,**kw:load_fused(*a,**options,**kw)
        sys.argv=[sys.argv[0],'--directory',str(root),'--output',str(out/'fused_benchmark'/name),'--mode',mode]
        fast_bench.main();return
    previous=json.loads((out/'manifest.json').read_text())
    old={**previous['code'],**previous['prior_unchanged']}
    for f,digest in old.items():assert sha(f)==digest,f
    files=[Path(__file__).parent/f for f in ['opt4_fused.py','opt4_fused_check.py','opt4_fused_suite.py']]
    manifest=dict(code={str(f.resolve()):sha(f)for f in files},prior_unchanged=old,variants=VARIANTS,
                  reference_for_bitwise_check='optimization4/quality; quality vs Opt2 is measured separately, not bitwise',assignments=ASSIGNMENTS)
    atomic_json(out/'fused_manifest.json',manifest)
    for stage,module,extra in [('fused_benchmark','migration.unified_router.opt4_fused_suite',['--worker']),
                                ('fused_validation','migration.unified_router.opt4_fused_check',[])]:
        update(out/'fused_status.json',status='running',stage=stage,pid=os.getpid());print(stage,flush=True)
        cmd=[str(PYTHON),'-u',str(ROOT/'migration/runtime/launch.py'),'--gpus','0,1,2,3,4,5,6,7','--module',module,
             '--logdir',str(out/f'{stage}_logs'),'--','--directory',str(root),*extra]
        with (out/f'{stage}_launcher.log').open('w')as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:update(out/'fused_status.json',status='failed',stage=stage);raise RuntimeError(stage)
    for f,digest in {**manifest['code'],**old}.items():assert sha(f)==digest,f
    checks=[json.loads((out/f'fused_validation/rank{i}.json').read_text())for i in range(8)]
    update(out/'fused_status.json',status='completed',stage='speed_and_regression_complete',all_bitwise_passed=all(r['passed']for r in checks))


if __name__=='__main__':main()
